import os
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import String, Integer, Text, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


# ================== APP ==================
app = FastAPI()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")
API_KEY = os.getenv("API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

DATABASE_URL = os.getenv("DATABASE_URL")

# Разрешаем запросы с фронта Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ret-ashy.vercel.app"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ================== DB (PostgreSQL) ==================
class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    # твой короткий id (uuid hex[:10])
    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)

    buyer_id: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    accepted_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


def _to_async_db_url(url: str) -> str:
    """
    Railway часто даёт:
      postgresql://...
    А SQLAlchemy async должен быть:
      postgresql+asyncpg://...
    """
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = None
SessionLocal = None

if DATABASE_URL:
    async_db_url = _to_async_db_url(DATABASE_URL)
    engine = create_async_engine(async_db_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    if SessionLocal is None:
        raise HTTPException(500, detail="DATABASE_URL not set in backend service variables")
    return SessionLocal()


@app.on_event("startup")
async def startup():
    """
    При старте создаём таблицы (если их нет).
    Для первого опыта это проще всего.
    """
    if engine is None:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ================== TELEGRAM ==================
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


# ================== API MODELS ==================
class OrderPayload(BaseModel):
    text: str
    buyer_id: int


@app.get("/health")
def health():
    return {"ok": True}


# ================== SETTINGS HELPERS ==================
PAYMENT_KEY = "payment_text"


async def get_payment_text(session: AsyncSession) -> str:
    row = await session.get(Setting, PAYMENT_KEY)
    return row.value if row else ""


async def set_payment_text(session: AsyncSession, text: str) -> None:
    row = await session.get(Setting, PAYMENT_KEY)
    if row:
        row.value = text
    else:
        session.add(Setting(key=PAYMENT_KEY, value=text))
    await session.commit()


# ================== CREATE ORDER (mini app -> backend -> group chat) ==================
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

    session = await get_session()
    async with session:
        session.add(Order(order_id=order_id, buyer_id=payload.buyer_id, text=text))
        await session.commit()

    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Принять заказ", "callback_data": f"accept:{order_id}"}
        ]]
    }

    await tg_call("sendMessage", {
        "chat_id": GROUP_CHAT_ID,
        "text": f"{text}\n\n🆔 Заказ: {order_id}",
        "reply_markup": keyboard
    })

    return {"ok": True, "order_id": order_id}


# ================== TELEGRAM WEBHOOK ==================
@app.post("/telegram/webhook")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
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
                await tg_call("sendMessage", {
                    "chat_id": chat_id,
                    "text": "⛔ Нет прав менять реквизиты."
                })
                return {"ok": True}

            new_text = text[len("/pay"):].strip()
            if not new_text:
                await tg_call("sendMessage", {
                    "chat_id": chat_id,
                    "text": "Напиши так:\n/pay\nКарта: ...\nСБП: ..."
                })
                return {"ok": True}

            session = await get_session()
            async with session:
                await set_payment_text(session, new_text)

            await tg_call("sendMessage", {
                "chat_id": chat_id,
                "text": "✅ Реквизиты сохранены."
            })
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

    # чтобы убрать "часики" на кнопке
    try:
        await tg_call("answerCallbackQuery", {"callback_query_id": cb_id})
    except Exception:
        pass

    if data.startswith("accept:"):
        order_id = data.split("accept:", 1)[1].strip()

        session = await get_session()
        async with session:
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}

            if order.accepted:
                return {"ok": True}

            accepter = (
                f"@{from_user.get('username')}"
                if from_user.get("username")
                else (from_user.get("first_name") or "Пользователь")
            )

            payment_text = await get_payment_text(session)

            order.accepted = True
            order.accepted_by = accepter
            await session.commit()

        if not payment_text:
            await tg_call("sendMessage", {
                "chat_id": GROUP_CHAT_ID,
                "text": f"⚠️ Заказ {order_id} принят: {accepter}\n"
                        f"Но реквизиты не заданы.\nАдмин: /pay <текст>"
            })
            return {"ok": True}

        await tg_call("sendMessage", {
            "chat_id": GROUP_CHAT_ID,
            "text": f"✅ Заказ {order_id} принят: {accepter}\n\n💳 Реквизиты:\n{payment_text}"
        })

        await tg_call("sendMessage", {
            "chat_id": order.buyer_id,
            "text": f"✅ Ваш заказ {order_id} принят.\n\n💳 Реквизиты для оплаты:\n{payment_text}"
        })

        if message_id:
            try:
                await tg_call("editMessageReplyMarkup", {
                    "chat_id": chat.get("id"),
                    "message_id": message_id,
                    "reply_markup": {
                        "inline_keyboard": [[{"text": "✅ Принято", "callback_data": "noop"}]]
                    }
                })
            except Exception:
                pass

    from sqlalchemy import text

    @app.get("/db/ping")
    async def db_ping():
        session = await get_session()
        async with session:
            result = await session.execute(text("SELECT 1"))
            return {"db": "ok", "value": result.scalar_one()}

    return {"ok": True}
