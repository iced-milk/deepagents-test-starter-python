async def handler(context):
    data = {
        "status": "ok",
        "runId": context.run_id,
        "env": context.env,
    }

    return data