"""Abort a specific active agent run by conversationId."""
import sys
from datetime import datetime, timezone


# ─── Unified logger with [stop][timestamp] prefix ───

class _Logger:
    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def log(*args: object) -> None:
        print(f"[stop][{_Logger._ts()}]", *args, flush=True)

    @staticmethod
    def error(*args: object) -> None:
        print(f"[stop][{_Logger._ts()}]", *args, file=sys.stderr, flush=True)


logger = _Logger()


async def handler(context):
    body = context.request.body or {}
    conversation_id = body.get("conversationId")
    logger.log("conversationId:", conversation_id)

    if not conversation_id:
        logger.error("Missing conversationId")
        return {"status_code": 400, "body": "Missing conversationId"}

    aborted = context.utils.abort_active_run(conversation_id)
    logger.log("abort_active_run result:", {"aborted": aborted})

    return {
        "status": "aborting" if aborted else "idle",
        "conversationId": conversation_id,
        "aborted": aborted,
    }
