import os
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")
API_KEY = os.getenv("API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"ok": True, "service": "tg-miniapp-backend"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/send")
async def api_send(payload: Payload, x_api_key: str | None = Header(default=None)):
    if not BOT_TOKEN or not GROUP_CHAT_ID:
        raise HTTPException(500, "BOT_TOKEN / GROUP_CHAT_ID not set")

    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Bad API key")

    text = payload.text.strip()
    if not text:
        raise HTTPException(400, "Empty text")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    body = {"chat_id": GROUP_CHAT_ID, "text": text}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=body)
        data = r.json()

    if not data.get("ok"):
        raise HTTPException(500, data)

    return {"ok": True}
