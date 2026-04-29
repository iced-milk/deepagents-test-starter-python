"""Abort a specific active agent run by conversationId.
"""

async def handler(context):
    body = context.request.body or {}
    conversation_id = body.get("conversationId")

    if not conversation_id:
        return {"status_code": 400, "body": "Missing conversationId"}

    aborted = context.agents.abort_active_run(conversation_id)

    return {
        "status": "aborting" if aborted else "idle",
        "conversationId": conversation_id,
        "aborted": aborted,
    }
