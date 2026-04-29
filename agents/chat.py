import asyncio
import sys
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
)
from deepagents import create_deep_agent

# ─── Unified logger with [chat][timestamp] prefix ───

class _Logger:
    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def log(*args: object) -> None:
        print(f"[chat][{_Logger._ts()}]", *args, flush=True)

    @staticmethod
    def error(*args: object) -> None:
        print(f"[chat][{_Logger._ts()}]", *args, file=sys.stderr, flush=True)


logger = _Logger()

# ─── Lazy-loaded singletons ───

_model = None
_agent = None

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions concisely and clearly."


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
    return _agent


async def handler(context):
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
        logger.error("error:", str(e))
        return {"status_code": 500, "body": {"error": str(e)}}

    # Race agent invocation against cancel signal. LangChain's ainvoke has no
    # signal param, so wrap it in a Task and wait() against the cancel event.
    # On cancel, cancel the task to propagate CancelledError down to httpx and
    # release the upstream LLM request, then re-raise so the runtime returns
    # 499 (matches ts chat.ts `throw error` behavior).
    invoke_task = asyncio.ensure_future(
        agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    )
    cancel_task = asyncio.ensure_future(context.request.signal.wait())
    try:
        done, _ = await asyncio.wait(
            {invoke_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            logger.log("aborted by user")
            invoke_task.cancel()
            try:
                await invoke_task
            except BaseException:
                pass
            raise asyncio.CancelledError()

        try:
            result = invoke_task.result()
        except Exception as e:
            logger.error("error:", str(e))
            return {"status_code": 500, "body": {"error": str(e)}}

        ai_content = result["messages"][-1].content
        logger.log("ai:", ai_content)
        return {"response": ai_content}
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
            try:
                await cancel_task
            except BaseException:
                pass
