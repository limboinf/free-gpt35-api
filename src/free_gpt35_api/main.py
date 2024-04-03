import random
import string
import httpx
import asyncio
import json
import ssl
import traceback
from time import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Constants for the server and API configuration
port = 3040
base_url = "https://chat.openai.com"
api_url = f"{base_url}/backend-api/conversation"
refresh_interval = 600000   # Interval to refresh token in ms
error_wait = 120000         # Wait time in ms after an error

# Initialize global variables to store the session token and device ID
token = None
oai_device_id = None

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom headers
custom_headers = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "application/json",
    "oai-language": "en-US",
    "origin": base_url,
    "pragma": "no-cache",
    "referer": base_url,
    "sec-ch-ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

# SSL context for ignoring SSL errors
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


async def wait(ms):
    await asyncio.sleep(ms / 1000.0)


def generate_completion_id(prefix="cmpl-"):
    characters = string.ascii_letters + string.digits
    return prefix + ''.join(random.choice(characters) for i in range(28))


async def get_new_session_id():
    global token, oai_device_id
    new_device_id = str(uuid4())
    async with httpx.AsyncClient(headers=custom_headers, verify=False) as client:
        response = await client.post(
            f"{base_url}/backend-anon/sentinel/chat-requirements",
            headers={"oai-device-id": new_device_id},
        )
    print(f"System: Successfully refreshed session ID and token. {'' if not token else '(Now it\'s ready to process requests)'}")
    oai_device_id = new_device_id
    token = response.json()["token"]


async def chunks_to_lines(chunks_async):
    previous = ""
    async for chunk in chunks_async:
        previous += chunk.decode() if isinstance(chunk, bytes) else chunk
        eol_index = 0
        while (eol_index := previous.find("\n")) >= 0:
            line = previous[:eol_index + 1].rstrip()
            if line == "data: [DONE]":
                break

            if line.startswith("data: "):
                yield line

            previous = previous[eol_index + 1:]


async def lines_to_messages(lines_async):
    async for line in lines_async:
        message = line[len("data: "):]
        yield message


async def stream_completion(data):
    async for message in lines_to_messages(chunks_to_lines(data)):
        yield message
        

@app.post("/v1/chat/completions")
async def handle_chat_completion(request: Request):
    body = await request.json()
    print("---------------------------")
    print(body)
    print("---------------------------")
    stream_enabled = body.get("stream", False)
    full_content = ''
    request_id = generate_completion_id("chatcmpl-")

    try:
        messages = body.get("messages", [])
        mapped_messages = [{
            "author": {"role": message["role"]},
            "content": {"content_type": "text", "parts": [message["content"]]},
        } for message in messages]

        async with httpx.AsyncClient(headers=custom_headers, verify=False) as client:
            response = await client.post(
                api_url,
                json={
                    "action": "next",
                    "messages": mapped_messages,
                    "parent_message_id": str(uuid4()),
                    "model": "text-davinci-002-render-sha",
                    "timezone_offset_min": -180,
                    "suggestions": [],
                    "history_and_training_disabled": True,
                    "conversation_mode": {"kind": "primary_assistant"},
                    "websocket_request_id": str(uuid4()),
                },
                headers={
                    "oai-device-id": oai_device_id,
                    "openai-sentinel-chat-requirements-token": token,
                },
                timeout=None,
            )

        if stream_enabled:
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(headers=headers, content=sse_response(request_id, response, messages))
        else:
            async for message in stream_completion(response.aiter_text()):
                parsed = json.loads(message)
                content = parsed.get('message', {}).get('content', {}).get('parts', [{}])[0] or ""
                if any(msg['content'] == content for msg in body['messages']):
                    continue

                full_content = content if len(content) > len(full_content) else full_content

            d = {
                "id": request_id,
                "created": int(time()),
                "model": "gpt-3.5-turbo",
                "object": "chat.completion",
                "choices": [
                    {
                        "logprobs": None,
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                    },
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            return JSONResponse(content=d)

    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"status": False, "error": {"message": str(e), "type": "invalid_request_error"}})


async def sse_response(request_id, response, messages):
    full_content = ''
    async for message in stream_completion(response.aiter_text()):
        parsed = json.loads(message)
        content = parsed.get('message', {}).get('content', {}).get('parts', [{}])[0] or ""
        if any(msg['content'] == content for msg in messages):
            continue

        response_chunk = {
                "id": request_id,
                "created": int(time()),
                "object": "chat.completion.chunk",
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "delta": {
                            "content": content.replace(full_content, ''),
                        },
                        "index": 0,
                        "finish_reason": None,
                    },
                ],
            }
        yield f"data: {json.dumps(response_chunk)}\n\n"
        full_content = content if len(content) > len(full_content) else full_content


async def refresh_session_id():
    while True:
        try:
            await get_new_session_id()
            await wait(refresh_interval)
        except Exception as e:
            print("Error refreshing session ID, retrying in 1 minute...")
            print("If this error persists, your country may not be supported yet.")
            print("If your country was the issue, please consider using a U.S. VPN.")
            await wait(error_wait)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(refresh_session_id())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)