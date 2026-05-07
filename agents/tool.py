import asyncio
import re
import sys
import time
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from deepagents import create_deep_agent

# ─── Unified logger with [tool][timestamp] prefix ───

class _Logger:
    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def log(*args: object) -> None:
        print(f"[tool][{_Logger._ts()}]", *args, flush=True)

    @staticmethod
    def error(*args: object) -> None:
        print(f"[tool][{_Logger._ts()}]", *args, file=sys.stderr, flush=True)


logger = _Logger()

# ─── Lazy-loaded singletons ───

_model = None
_agent = None

SYSTEM_PROMPT = (
    f"You are a helpful assistant. Today's date is "
    f"{datetime.now(timezone.utc).date().isoformat()}. "
    "Use `internet_search` to look up information before answering. "
    "When searching, prefer including the current year or recent time range "
    "to get the latest results. Answer concisely."
)

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

GAP_THRESHOLD_MS = 3000
HEARTBEAT_INTERVAL_S = 5.0

# ─── Internet search tool ───
# Note: `name`/`max_results`/`output_format` directly mirror the ts side's
# DDGS({timeout:15000}) + tool(name="internet_search", maxResults=3).
internet_search = DuckDuckGoSearchResults(
    name="internet_search",
    max_results=3,
    output_format="list",
)


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
            tools=[internet_search],
            middleware=[
                ModelRetryMiddleware(max_retries=3),
                ModelCallLimitMiddleware(run_limit=30),
                ToolRetryMiddleware(max_retries=2, tools=["internet_search"]),
                ToolCallLimitMiddleware(tool_name="internet_search", run_limit=15),
            ],
        )
    else:
        logger.log("Agent already initialized, reusing")
    return _agent


async def _event_stream(agent, user_message: str, utils):
    """Stream the agent run as SSE frames for the chat UI.

    Consumes ``agent.astream(..., stream_mode="messages", version="v2")`` and
    translates the raw token stream into three semantic frame types that
    mirror the TypeScript counterpart in ``tool.ts``:

      - ``tool_call``   : a new tool invocation begins (name only; streamed
                          args fragments are logged but not forwarded).
      - ``tool_result`` : a ``ToolMessage`` carrying the tool's output
                          (content truncated to 500 chars).
      - ``ai_response`` : plain text tokens from the model's final answer,
                          with runs of 3+ newlines folded to a blank line.

    Termination contract:
      - Normal finish or caught business error → yield a final ``[DONE]`` frame.
      - Client cancel (outer handler calls ``aclose()``) → propagate
        ``GeneratorExit`` without yielding anything extra, so the generator
        exits cleanly and the upstream LLM HTTP request is released.
    """
    try:
        logger.log(f'starting stream for message: "{user_message[:80]}"')
        last_tick = time.monotonic()
        last_chunk_kind = "other"

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="messages",
            version="v2",
        ):
            gap_ms = int((time.monotonic() - last_tick) * 1000)
            if gap_ms > GAP_THRESHOLD_MS:
                # Annotate which chunk kind preceded the gap, so we can quickly
                # locate stalls (common case: tool_call → tool_result delay
                # caused by DDG search latency).
                logger.log(f"[gap] {gap_ms}ms before next chunk (after={last_chunk_kind})")
            last_tick = time.monotonic()

            # v2 stream_mode="messages" yields flat dicts: {"type","ns","data"}
            if chunk.get("type") != "messages":
                continue

            token, _metadata = chunk["data"]

            # 1) Streaming tool calls (AIMessageChunk with tool_call_chunks)
            tool_call_chunks = getattr(token, "tool_call_chunks", None)
            if isinstance(token, AIMessageChunk) and tool_call_chunks:
                last_chunk_kind = "tool_call"
                for tc in tool_call_chunks:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                    if name:
                        logger.log(f"tool call: {name}")
                        yield utils.sse({"type": "tool_call", "name": name})
                    if args:
                        logger.log(f"tool call args: {args}")
                continue

            # 2) Tool results (ToolMessage)
            if isinstance(token, ToolMessage):
                last_chunk_kind = "tool_result"
                # ToolMessage.content may be str | list; coerce to str for logging/SSE.
                text = token.content if isinstance(token.content, str) else str(token.content)
                logger.log(f"tool result [{token.name}]: {text[:150]}")
                yield utils.sse({
                    "type": "tool_result",
                    "name": token.name,
                    "content": text[:500],
                })
                continue

            # 3) AI text response (AIMessageChunk text)
            # Use `.text` (property) to mirror ts's `message.text`: it handles
            # both str content and list-of-parts content (multi-modal) uniformly,
            # extracting only the textual portion.
            if isinstance(token, AIMessageChunk):
                text = token.text
                if text:
                    last_chunk_kind = "ai_response"
                    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", text)
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
