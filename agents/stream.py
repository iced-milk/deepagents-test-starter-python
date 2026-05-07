import asyncio
import re
import sys
import time
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
)
from deepagents import create_deep_agent

# ─── Unified logger with [stream][timestamp] prefix ───

class _Logger:
    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def log(*args: object) -> None:
        print(f"[stream][{_Logger._ts()}]", *args, flush=True)

    @staticmethod
    def error(*args: object) -> None:
        print(f"[stream][{_Logger._ts()}]", *args, file=sys.stderr, flush=True)


logger = _Logger()

# ─── Lazy-loaded singletons ───

_model = None
_agent = None

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions concisely and clearly."

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

GAP_THRESHOLD_MS = 3000
HEARTBEAT_INTERVAL_S = 5.0


def get_env(context_env) -> dict[str, str]:
    source = context_env or {}
    required = ("AI_GATEWAY_API_KEY", "AI_GATEWAY_BASE_URL")
    missing = [k for k in required if not (source.get(k) or "").strip()]

    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    return {k: source[k] for k in required}


def get_model(env: dict[str, str]):
    global _model
    if _model is None:
        logger.log("Initializing model...")
        _model = init_chat_model(
            model="@Pages/deepseek-v4-flash",
            api_key=env["AI_GATEWAY_API_KEY"],
            base_url=env["AI_GATEWAY_BASE_URL"],
            model_provider="openai",
            temperature=0,
            timeout=300,
        )
    else:
        logger.log("Model already initialized, reusing")
    return _model


def get_agent(model):
    global _agent
    if _agent is None:
        logger.log("Initializing agent...")
        _agent = create_deep_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                ModelRetryMiddleware(max_retries=3),
                ModelCallLimitMiddleware(run_limit=30),
            ],
        )
    else:
        logger.log("Agent already initialized, reusing")
    return _agent


async def _event_stream(agent, user_message: str, utils):
    """Stream the agent run as SSE frames for the chat UI.

    Consumes ``agent.astream(..., stream_mode="messages", version="v2")`` and
    forwards every AI text token as an ``ai_response`` frame. Multi-blank-line
    runs in the model output are collapsed to a single blank line before emit.

    Termination contract:
      - Normal finish or caught business error → yield a final ``[DONE]`` frame.
      - Client cancel (outer handler calls ``aclose()``) → propagate
        ``GeneratorExit`` without yielding anything extra, so the generator
        exits cleanly and the upstream LLM HTTP request is released.
    """
    try:
        logger.log(f'starting stream for message: "{user_message[:80]}"')
        last_tick = time.monotonic()

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="messages",
            version="v2",
        ):
            gap_ms = int((time.monotonic() - last_tick) * 1000)
            if gap_ms > GAP_THRESHOLD_MS:
                logger.log(f"[gap] {gap_ms}ms before next chunk")
            last_tick = time.monotonic()

            # v2 format yields flat dicts: {"type", "ns", "data"}
            if chunk.get("type") != "messages":
                continue

            token, _metadata = chunk["data"]
            content = getattr(token, "content", None)
            if not content:
                continue

            cleaned = _MULTI_NEWLINE_RE.sub("\n\n", content)
            if cleaned:
                logger.log("ai response:", cleaned)
                yield utils.sse({"type": "ai_response", "content": cleaned})
        logger.log("stream completed")
        yield utils.sse("[DONE]")
    except Exception as e:
        logger.error("error:", str(e))
        yield utils.sse({"type": "error_message", "content": f"Stream error: {str(e)}"})
        yield utils.sse("[DONE]")


async def handler(context):
    logger.log(
        "conversationId:", getattr(context, "conversation_id", None),
        "runId:", getattr(context, "run_id", None),
    )
    body = context.request.body or {}
    message = body.get("message")
    logger.log("user message:", message)

    if not message:
        logger.error("Missing chat message")
        return {"status_code": 400, "body": "Missing chat message"}

    try:
        env = get_env(context.env)
        model = get_model(env)
        agent = get_agent(model)
    except Exception as e:
        msg = str(e)
        logger.error(msg)
        return {"status_code": 500, "body": {"error": msg}}

    async def gen():
        # Race three sources in a single asyncio.wait:
        #   - pending     : next frame from _event_stream
        #   - cancel_task : context.request.signal (set by runtime on /stop)
        #   - timeout     : heartbeat window; emit a ping on timeout
        # On cancel, break out and let the finally block aclose() the generator,
        # which propagates GeneratorExit down to agent.astream → httpx.
        agen = _event_stream(agent, message, context.utils).__aiter__()
        cancel_task = asyncio.ensure_future(context.request.signal.wait())
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(agen.__anext__())

                done, _ = await asyncio.wait(
                    {pending, cancel_task},
                    timeout=HEARTBEAT_INTERVAL_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancel_task in done:
                    logger.log("cancel signal received; aborting stream")
                    break

                if not done:
                    ts = int(time.time() * 1000)
                    logger.log(f"[heartbeat] ping {ts}")
                    yield context.utils.sse({"type": "ping", "ts": ts})
                    continue

                try:
                    frame = pending.result()
                except StopAsyncIteration:
                    break
                pending = None
                yield frame
        finally:
            # Settle pending __anext__ before aclose(); a running async generator
            # rejects aclose() with "asynchronous generator is already running".
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except BaseException:
                    pass
            if not cancel_task.done():
                cancel_task.cancel()
                try:
                    await cancel_task
                except BaseException:
                    pass
            # Inject GeneratorExit into _event_stream → agent.astream → httpx,
            # releasing the upstream LLM request. Closest Python equivalent of
            # ts AbortSignal.
            try:
                await agen.aclose()
            except Exception as e:
                logger.error("agen.aclose error:", str(e))

    return context.utils.stream_sse(gen())
