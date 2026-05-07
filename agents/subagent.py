import asyncio
import sys
import time
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
)
from langchain_core.messages import AIMessageChunk, ToolMessage
from deepagents import SubAgent, create_deep_agent

# ─── Unified logger with [subagent][timestamp] prefix ───


class _Logger:
    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def log(*args: object) -> None:
        print(f"[subagent][{_Logger._ts()}]", *args, flush=True)

    @staticmethod
    def error(*args: object) -> None:
        print(f"[subagent][{_Logger._ts()}]", *args, file=sys.stderr, flush=True)


logger = _Logger()

# ─── Lazy-loaded singletons ───

_model = None
_agent = None

SYSTEM_PROMPT = (
    "You are a coordinator. Always delegate research tasks to your researcher "
    "subagent using the task tool. Keep your final response to one sentence."
)

GAP_THRESHOLD_MS = 3000
HEARTBEAT_INTERVAL_S = 5.0

# Research subagent definition — mirrors researchSubagent in subagent.ts.
research_subagent: SubAgent = {
    "name": "research-agent",
    "description": "Researches topics thoroughly",
    "system_prompt": (
        "You are a thorough researcher. Research the given topic and provide "
        "a concise summary in 2-3 sentences."
    ),
    "middleware": [
        ModelRetryMiddleware(max_retries=3),
        ModelCallLimitMiddleware(run_limit=30),
    ],
}


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
        logger.log("Initializing agent with subagents...")
        _agent = create_deep_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            subagents=[research_subagent],
            middleware=[
                ModelRetryMiddleware(max_retries=3),
                ModelCallLimitMiddleware(run_limit=30),
            ],
        )
    else:
        logger.log("Agent already initialized, reusing")
    return _agent


async def _event_stream(agent, user_message: str, utils):
    """Stream a multi-agent run as SSE frames.

    Consumes ``agent.astream(stream_mode=["updates", "messages"], subgraphs=True,
    version="v2")``. Each chunk is ``{type, ns, data}`` where ``ns`` is ``()``
    for the main agent and ``("tools:<pregel_id>",)`` for a spawned subagent.

    Emitted frames:
      - ``subagent_lifecycle`` (pending/complete): read from the main agent's
        ``task`` tool_call args and ToolMessage — always reliable.
      - ``source_switch`` / ``ai_response``: token-level output routed per
        namespace, keyed by the 8-char short id of the pregel task for
        per-subagent grouping.

    We deliberately do not map ``tools:<pregel_id>`` back to a readable subagent
    name (e.g. "research-agent") — the Pregel task id is unrelated to the
    tool_call_id, so the pairing would be a fragile arrival-order heuristic.
    """
    seen_namespaces: set[str] = set()
    current_source = ""

    def short_id(ns_str: str) -> str:
        return ns_str.split(":", 1)[-1][:8] if ns_str else ""

    try:
        logger.log(f'starting combined stream for message: "{user_message[:80]}"')
        last_tick = time.monotonic()

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ):
            mode = chunk.get("type")
            ns = chunk.get("ns") or ()
            data = chunk.get("data")

            gap_ms = int((time.monotonic() - last_tick) * 1000)
            if gap_ms > GAP_THRESHOLD_MS:
                ns_label = "/".join(ns) if ns else "main"
                logger.log(f"[gap] {gap_ms}ms before next chunk (ns={ns_label}, mode={mode})")
            last_tick = time.monotonic()

            is_subagent = any(s.startswith("tools:") for s in ns)
            subagent_ns = next((s for s in ns if s.startswith("tools:")), None) if is_subagent else None

            if mode == "updates":
                if not isinstance(data, dict):
                    continue
                for node_name, node_data in data.items():
                    messages = (node_data or {}).get("messages", []) if isinstance(node_data, dict) else []

                    # Main agent dispatched a task tool_call → subagent pending.
                    if not ns and node_name == "model_request":
                        had_task = False
                        for msg in messages:
                            for tc in getattr(msg, "tool_calls", []) or []:
                                if tc.get("name") == "task":
                                    had_task = True
                                    args = tc.get("args") or {}
                                    subagent_type = args.get("subagent_type", "subagent")
                                    description = (args.get("description") or "")[:200]
                                    tc_id = tc.get("id")
                                    logger.log(
                                        f'[lifecycle] PENDING  → subagent "{subagent_type}" ({tc_id})'
                                    )
                                    logger.log(f"  description: {description or 'N/A'}")
                                    yield utils.sse({
                                        "type": "subagent_lifecycle",
                                        "status": "pending",
                                        "subagent_type": subagent_type,
                                        "tool_call_id": tc_id,
                                        "description": description,
                                    })
                        if not had_task:
                            logger.log(f"[updates] [main agent] step: {node_name}")

                    # First sight of a subagent namespace — log only.
                    if subagent_ns:
                        if subagent_ns not in seen_namespaces:
                            seen_namespaces.add(subagent_ns)
                            logger.log(
                                f'[lifecycle] RUNNING  → subagent namespace {subagent_ns} '
                                f"(short id: {short_id(subagent_ns)})"
                            )
                        logger.log(f"[updates] [{subagent_ns}] step: {node_name}")

                    # Main agent's tools node returned a task ToolMessage → complete.
                    if not ns and node_name == "tools":
                        for msg in messages:
                            if isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
                                if getattr(msg, "name", None) != "task":
                                    continue
                                tc_id = getattr(msg, "tool_call_id", None)
                                content = getattr(msg, "content", "")
                                content_str = content if isinstance(content, str) else str(content)
                                logger.log(f"[lifecycle] COMPLETE → subagent tool_call {tc_id}")
                                logger.log(f"  Result preview: {content_str[:200]}...")
                                yield utils.sse({
                                    "type": "subagent_lifecycle",
                                    "status": "complete",
                                    "tool_call_id": tc_id,
                                    "content": content_str[:500],
                                })

                    if not ns and node_name not in ("model_request", "tools"):
                        logger.log(f"[updates] [main agent] step: {node_name}")

            elif mode == "messages":
                if not isinstance(data, tuple) or len(data) < 1:
                    continue
                token = data[0]

                if is_subagent and subagent_ns:
                    sid = short_id(subagent_ns)
                    if subagent_ns != current_source:
                        current_source = subagent_ns
                        logger.log(f"--- [subagent {sid} ({subagent_ns})] ---")
                        yield utils.sse({
                            "type": "source_switch",
                            "agent": "subagent",
                            "subagent_id": sid,
                            "namespace": subagent_ns,
                        })
                    if isinstance(token, AIMessageChunk):
                        text = token.text
                        if text and not token.tool_call_chunks:
                            logger.log(f"[token] [subagent {sid}] {text}")
                            yield utils.sse({
                                "type": "ai_response",
                                "content": text,
                                "agent": "subagent",
                                "subagent_id": sid,
                                "namespace": subagent_ns,
                            })
                else:
                    if current_source != "main":
                        current_source = "main"
                        logger.log("--- [main agent] ---")
                        yield utils.sse({"type": "source_switch", "agent": "main"})
                    if isinstance(token, AIMessageChunk):
                        text = token.text
                        if text and not token.tool_call_chunks:
                            logger.log(f"[token] [main] {text}")
                            yield utils.sse({
                                "type": "ai_response",
                                "content": text,
                                "agent": "main",
                            })

        logger.log("stream completed")
    except Exception as e:
        logger.error("stream error:", str(e))
        yield utils.sse({
            "type": "error_message",
            "content": f"Stream error: {str(e)}",
        })

    if seen_namespaces:
        logger.log("--- Subagent namespaces seen this run ---")
        for ns_str in seen_namespaces:
            logger.log(f"  {ns_str} (short id: {short_id(ns_str)})")

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
            # releasing the upstream LLM request.
            try:
                await agen.aclose()
            except Exception as e:
                logger.error("agen.aclose error:", str(e))

    return context.utils.stream_sse(gen())
