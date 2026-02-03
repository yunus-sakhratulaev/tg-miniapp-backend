import os
import uuid
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")
API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

ORDERS: dict[str, int] = {}     # order_id -> buyer_id
ACCEPTED: set[str] = set()
PAYMENT_TEXT: str = ""          # реквизиты задаёт админ

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ret-ashy.vercel.app"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class OrderPayload(BaseModel):
    text: str
    buyer_id: int

async def tg_call(method: str, payload: dict):
    if not BOT_TOKEN:
        raise HTTPException(500, detail="BOT_TOKEN not set")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload)
        data = r.json()
    if not data.get("ok"):
        raise HTTPException(500, detail=data)
    return data["result"]

@app.get("/health")
def health():
    return {"ok": True}

# Создание заказа из mini-app
@app.post("/api/order")
async def create_order(payload: OrderPayload, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, detail="Bad API key")
    if not GROUP_CHAT_ID:
        raise HTTPException(500, detail="GROUP_CHAT_ID not set")

    text = payload.text.strip()
    if not text:
        raise HTTPException(400, detail="Empty text")

    order_id = uuid.uuid4().hex[:10]
    ORDERS[order_id] = payload.buyer_id

    keyboard = {"inline_keyboard": [[
        {"text": "✅ Принять заказ", "callback_data": f"accept:{order_id}"}
    ]]}

    await tg_call("sendMessage", {
        "chat_id": GROUP_CHAT_ID,
        "text": f"{text}\n\n🆔 Заказ: {order_id}",
        "reply_markup": keyboard
    })

    return {"ok": True, "order_id": order_id}

# Webhook: ловим /pay и кнопки accept
@app.post("/telegram/webhook")
async def telegram_webhook(req: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(401, detail="Bad webhook secret")

    update = await req.json()

    # 1) Админ задаёт реквизиты: /pay <текст>
    msg = update.get("message")
    if msg:
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        from_id = msg.get("from", {}).get("id")

        if GROUP_CHAT_ID and chat_id == int(GROUP_CHAT_ID) and text.startswith("/pay"):
            if from_id not in ADMIN_IDS:
                await tg_call("sendMessage", {"chat_id": chat_id, "text": "⛔ Нет прав менять реквизиты."})
                return {"ok": True}

            new_text = text[len("/pay"):].strip()
            if not new_text:
                await tg_call("sendMessage", {
                    "chat_id": chat_id,
                    "text": "Напиши так:\n/pay\nКарта: ...\nСБП: ..."})
                return {"ok": True}

            global PAYMENT_TEXT
            PAYMENT_TEXT = new_text
            await tg_call("sendMessage", {"chat_id": chat_id, "text": "✅ Реквизиты сохранены."})
            return {"ok": True}

        return {"ok": True}

    # 2) Нажатие “Принять заказ”
    cb = update.get("callback_query")
    if not cb:
        return {"ok": True}

    cb_id = cb.get("id")
    data = cb.get("data", "")
    from_user = cb.get("from", {})
    message = cb.get("message", {})
    message_id = message.get("message_id")
    chat = message.get("chat", {})

    try:
        await tg_call("answerCallbackQuery", {"callback_query_id": cb_id})
    except Exception:
        pass

    if data.startswith("accept:"):
        order_id = data.split("accept:", 1)[1].strip()
        buyer_id = ORDERS.get(order_id)
        if not buyer_id:
            return {"ok": True}

        if order_id in ACCEPTED:
            return {"ok": True}
        ACCEPTED.add(order_id)

        accepter = f"@{from_user.get('username')}" if from_user.get("username") else (from_user.get("first_name") or "Пользователь")

        if not PAYMENT_TEXT:
            await tg_call("sendMessage", {
                "chat_id": GROUP_CHAT_ID,
                "text": f"⚠️ Заказ {order_id} принят: {accepter}\nНо реквизиты не заданы.\nАдмин: /pay <текст>"})
            return {"ok": True}

        await tg_call("sendMessage", {
            "chat_id": GROUP_CHAT_ID,
            "text": f"✅ Заказ {order_id} принят: {accepter}\n\n💳 Реквизиты:\n{PAYMENT_TEXT}"
        })

        await tg_call("sendMessage", {
            "chat_id": buyer_id,
            "text": f"✅ Ваш заказ {order_id} принят.\n\n💳 Реквизиты для оплаты:\n{PAYMENT_TEXT}"
        })

        if message_id:
            try:
                await tg_call("editMessageReplyMarkup", {
                    "chat_id": chat.get("id"),
                    "message_id": message_id,
                    "reply_markup": {"inline_keyboard": [[{"text": "✅ Принято", "callback_data": "noop"}]]}
                })
            except Exception:
                pass

    return {"ok": True}
