async def handler(context):
    data = {
        "status": "ok",
        "conversationId": context.conversation_id,
        "runId": context.run_id,
        "env": context.env,
    }

    return data