# v1 Готовый вариант
import os
import uuid
import hmac
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    String,
    Integer,
    BigInteger,
    DateTime,
    Text,
    JSON,
    Boolean,
    select,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")  # -100...
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

RESET_DB = os.getenv("RESET_DB", "0") == "1"

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


# =========================
# DB Models
# =========================
class Base(DeclarativeBase):
    pass


class OrderStatus(str, Enum):
    NEW = "NEW"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    RECEIPT_SENT = "RECEIPT_SENT"
    PAID = "PAID"
    CANCELLED = "CANCELLED"


class PaymentMethod(Base):
    __tablename__ = "payment_methods"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text(), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)

    # ✅ Telegram user_id часто больше int32 -> только BIGINT
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    items: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    address_text: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    delivery_slot: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=OrderStatus.NEW.value)

    payment_method_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ✅ тоже BIGINT на всякий случай
    accepted_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    receipt_file_id: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    receipt_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# =========================
# DB Engine
# =========================
engine = create_async_engine(DATABASE_URL, echo=False) if DATABASE_URL else None
SessionLocal: Optional[async_sessionmaker[AsyncSession]] = (
    async_sessionmaker(engine, expire_on_commit=False) if engine else None
)

DB_READY = False
DB_ERROR: str = ""


# =========================
# APP
# =========================
app = FastAPI()

# CORS: чтобы Vercel+Telegram WebView не мешали
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Utils
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def require_db():
    if not SessionLocal or not DB_READY:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {DB_ERROR or 'not ready'}")


async def tg_call(method: str, payload: dict[str, Any]):
    if not BOT_TOKEN:
        raise HTTPException(500, detail="BOT_TOKEN not set")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, json=payload)
        data = r.json()
    if not data.get("ok"):
        raise HTTPException(500, detail={"telegram_error": data, "method": method})
    return data["result"]


def verify_webapp_init_data(init_data: str) -> dict[str, str]:
    if not BOT_TOKEN:
        raise HTTPException(500, detail="BOT_TOKEN not set (needed for initData validation)")
    if not init_data:
        raise HTTPException(401, detail="Missing initData")

    from urllib.parse import parse_qsl
    pairs = parse_qsl(init_data, keep_blank_values=True)
    data_map: dict[str, str] = dict(pairs)

    received_hash = data_map.pop("hash", None)
    if not received_hash:
        raise HTTPException(401, detail="Bad initData: missing hash")

    data_check_string = "\n".join([f"{k}={data_map[k]}" for k in sorted(data_map.keys())])

    secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode("utf-8"), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        raise HTTPException(401, detail="Bad initData signature")

    return data_map


def extract_user_id_from_init_data_map(data_map: dict[str, str]) -> int:
    user_raw = data_map.get("user")
    if not user_raw:
        return 0
    import json
    try:
        user_json = json.loads(user_raw)
        return int(user_json.get("id", 0))
    except Exception:
        return 0


def format_order_for_admin(order: Order) -> str:
    lines = []
    lines.append("🛒 *Новый заказ*")
    lines.append(f"🆔 Заказ: `{order.id}`")
    lines.append(f"👤 User ID: `{order.user_id}`")
    lines.append("")
    lines.append("*Состав:*")

    items = (order.items or {}).get("items") or []
    for it in items:
        title = it.get("title", "Товар")
        qty = it.get("qty", 0)
        price = it.get("price", 0)
        s = it.get("sum", qty * price)
        lines.append(f"• {title} — {qty} × {price} ₽ = {s} ₽")

    lines.append("")
    lines.append(f"*ИТОГО: {order.total} ₽*")
    return "\n".join(lines)


# =========================
# Schemas
# =========================
class OrderItemIn(BaseModel):
    product_id: int
    title: str
    price: int
    qty: int = Field(ge=1)


class CreateOrderIn(BaseModel):
    initData: str
    items: list[OrderItemIn]
    delivery_slot: Optional[str] = None
    address_text: Optional[str] = None


class CreateOrderOut(BaseModel):
    ok: bool = True
    order_id: str
    total: int


# =========================
# Startup: AUTO RESET (если RESET_DB=1)
# =========================
@app.on_event("startup")
async def _startup():
    global DB_READY, DB_ERROR

    if not engine:
        DB_READY = False
        DB_ERROR = "DATABASE_URL not set"
        return

    try:
        async with engine.begin() as conn:
            if RESET_DB:
                # ⚠️ УДАЛЯЕТ ТАБЛИЦЫ. ДАННЫЕ ПРОПАДУТ.
                await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

        # seed payment methods
        async with SessionLocal() as session:
            res = await session.execute(select(PaymentMethod).where(PaymentMethod.is_active == True))
            methods = res.scalars().all()
            if not methods:
                now = utcnow()
                session.add_all([
                    PaymentMethod(title="Карта 1", text="Карта: 0000 0000 0000 0000\nПолучатель: ...", is_active=True, created_at=now),
                    PaymentMethod(title="СБП", text="СБП по номеру: +7...\nБанк: ...\nПолучатель: ...", is_active=True, created_at=now),
                ])
                await session.commit()

        DB_READY = True
        DB_ERROR = ""
    except Exception as e:
        DB_READY = False
        DB_ERROR = repr(e)


@app.get("/health")
def health():
    return {"ok": True, "db_ready": DB_READY, "db_error": DB_ERROR, "reset_db": RESET_DB}


# =========================
# API
# =========================
@app.post("/api/order", response_model=CreateOrderOut)
async def create_order(payload: CreateOrderIn):
    require_db()

    if not GROUP_CHAT_ID:
        raise HTTPException(500, detail="GROUP_CHAT_ID not set")
    if not payload.items:
        raise HTTPException(400, detail="Empty items")

    data_map = verify_webapp_init_data(payload.initData)
    user_id = extract_user_id_from_init_data_map(data_map)
    if not user_id:
        raise HTTPException(401, detail="Cannot determine user_id from initData")

    total = sum(it.qty * it.price for it in payload.items)
    order_id = uuid.uuid4().hex[:10]
    now = utcnow()

    items_struct = {
        "items": [
            {"product_id": it.product_id, "title": it.title, "price": it.price, "qty": it.qty, "sum": it.qty * it.price}
            for it in payload.items
        ]
    }

    async with SessionLocal() as session:
        order = Order(
            id=order_id,
            user_id=user_id,
            items=items_struct,
            address_text=None,
            delivery_slot=payload.delivery_slot,
            total=total,
            status=OrderStatus.NEW.value,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.commit()

        keyboard = {
            "inline_keyboard": [[
                {"text": "💳 Выбрать реквизиты", "callback_data": f"choosepay:{order_id}"},
                {"text": "❌ Отменить", "callback_data": f"cancel:{order_id}"},
            ]]
        }

        # Если тут упадёт — увидишь нормальную 500 (уже не из-за БД)
        await tg_call("sendMessage", {
            "chat_id": int(GROUP_CHAT_ID),
            "text": format_order_for_admin(order),
            "parse_mode": "Markdown",
            "reply_markup": keyboard,
        })

    return CreateOrderOut(order_id=order_id, total=total)


# =========================
# Webhook placeholder (если у тебя он в другом файле — оставь как было)
# =========================
@app.post("/telegram/webhook")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(401, detail="Bad webhook secret")
    return {"ok": True}
