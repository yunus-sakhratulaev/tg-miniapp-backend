import os
import uuid
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
    Float,
    JSON,
    select,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")  # Telegram group id (e.g. -100123...)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

DATABASE_URL = os.getenv("DATABASE_URL")  # Railway Postgres provides this
# SQLAlchemy async требует asyncpg драйвер
# Railway обычно даёт postgres://... или postgresql://...
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Магазин (координаты) и формула доставки
SHOP_LAT = float(os.getenv("SHOP_LAT", "55.751244"))  # дефолт: Москва центр
SHOP_LNG = float(os.getenv("SHOP_LNG", "37.618423"))
DELIVERY_BASE = int(os.getenv("DELIVERY_BASE", "0"))         # базовая цена доставки
DELIVERY_PER_KM = int(os.getenv("DELIVERY_PER_KM", "80"))    # цена за км
DELIVERY_MIN = int(os.getenv("DELIVERY_MIN", "0"))
DELIVERY_MAX = int(os.getenv("DELIVERY_MAX", "2000"))

# CORS: Vercel домен mini-app
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "https://ret-ashy.vercel.app").split(",")


# =========================
# DB
# =========================
class Base(DeclarativeBase):
    pass


class OrderStatus(str, Enum):
    NEW = "NEW"                        # создан, в группу отправлен
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    RECEIPT_SENT = "RECEIPT_SENT"
    PAID = "PAID"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text(), nullable=False)


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # короткий id (hex)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # данные заказа
    items: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    address_text: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    delivery_slot: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    items_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delivery_price: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default=OrderStatus.NEW.value)

    accepted_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    payment_deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    receipt_file_id: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    receipt_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


engine = create_async_engine(DATABASE_URL, echo=False) if DATABASE_URL else None
SessionLocal: Optional[async_sessionmaker[AsyncSession]] = (
    async_sessionmaker(engine, expire_on_commit=False) if engine else None
)


# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =========================
# Utils
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def tg_call(method: str, payload: dict[str, Any]):
    if not BOT_TOKEN:
        raise HTTPException(500, detail="BOT_TOKEN not set")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload)
        data = r.json()
    if not data.get("ok"):
        raise HTTPException(500, detail=data)
    return data["result"]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calc_delivery_price(lat: Optional[float], lng: Optional[float]) -> tuple[int, Optional[float]]:
    if lat is None or lng is None:
        return 0, None
    km = _haversine_km(SHOP_LAT, SHOP_LNG, lat, lng)
    price = int(round(DELIVERY_BASE + km * DELIVERY_PER_KM))
    price = max(DELIVERY_MIN, min(DELIVERY_MAX, price))
    return price, km


def verify_webapp_init_data(init_data: str) -> dict[str, str]:
    """
    Проверка Telegram WebApp initData.
    Если ок — возвращает map параметров.
    """
    if not BOT_TOKEN:
        raise HTTPException(500, detail="BOT_TOKEN not set (needed for initData validation)")
    if not init_data:
        raise HTTPException(401, detail="Missing initData")

    pairs = [p for p in init_data.split("&") if p]
    data_map: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        data_map[k] = v

    received_hash = data_map.pop("hash", None)
    if not received_hash:
        raise HTTPException(401, detail="Bad initData: missing hash")

    data_check_string = "\n".join([f"{k}={data_map[k]}" for k in sorted(data_map.keys())])

    secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode("utf-8"), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        raise HTTPException(401, detail="Bad initData signature")

    auth_date = data_map.get("auth_date")
    if auth_date and auth_date.isdigit():
        dt = datetime.fromtimestamp(int(auth_date), tz=timezone.utc)
        if utcnow() - dt > timedelta(hours=24):
            raise HTTPException(401, detail="initData expired")

    return data_map


async def get_setting(session: AsyncSession, key: str, default: str = "") -> str:
    row = await session.get(Setting, key)
    return row.value if row else default


async def set_setting(session: AsyncSession, key: str, value: str) -> None:
    row = await session.get(Setting, key)
    if row:
        row.value = value
    else:
        session.add(Setting(key=key, value=value))
    await session.commit()


def format_order_for_admin(order: Order) -> str:
    lines = []
    lines.append("🛒 *Новый заказ*")
    lines.append(f"🆔 Заказ: `{order.id}`")
    lines.append(f"👤 User ID: `{order.user_id}`")
    if order.address_text:
        lines.append(f"📍 Адрес: {order.address_text}")
    if order.lat is not None and order.lng is not None:
        lines.append(f"🧭 Координаты: {order.lat:.6f}, {order.lng:.6f}")
    if order.delivery_slot:
        lines.append(f"🕒 Время доставки: {order.delivery_slot}")
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
    lines.append(f"Товары: *{order.items_total} ₽*")
    lines.append(f"Доставка: *{order.delivery_price} ₽*")
    lines.append(f"*ИТОГО: {order.total} ₽*")
    return "\n".join(lines)


def format_payment_to_user(order_id: str, payment_text: str) -> str:
    return (
        f"✅ Ваш заказ `{order_id}` принят.\n\n"
        f"💳 *Реквизиты для оплаты:*\n{payment_text}\n\n"
        f"⏳ У вас *15 минут* на оплату.\n"
        f"После оплаты отправьте сюда *фото/файл чека*."
    )


# =========================
# Schemas
# =========================
class DeliveryQuoteIn(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None
    address_text: Optional[str] = None


class DeliveryQuoteOut(BaseModel):
    ok: bool = True
    price: int
    distance_km: Optional[float] = None


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
    lat: Optional[float] = None
    lng: Optional[float] = None


class CreateOrderOut(BaseModel):
    ok: bool = True
    order_id: str
    total: int
    delivery_price: int


# =========================
# Startup
# =========================
@app.on_event("startup")
async def _startup():
    if not engine:
        raise RuntimeError("DATABASE_URL not set")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
def health():
    return {"ok": True}


# =========================
# API (Mini App -> Backend)
# =========================
@app.post("/api/delivery/quote", response_model=DeliveryQuoteOut)
async def delivery_quote(payload: DeliveryQuoteIn):
    price, km = calc_delivery_price(payload.lat, payload.lng)
    return DeliveryQuoteOut(price=price, distance_km=km)


@app.post("/api/order", response_model=CreateOrderOut)
async def create_order(payload: CreateOrderIn):
    if not GROUP_CHAT_ID:
        raise HTTPException(500, detail="GROUP_CHAT_ID not set")
    if not SessionLocal:
        raise HTTPException(500, detail="DB not configured")
    if not payload.items:
        raise HTTPException(400, detail="Empty items")

    # verify initData signature (auth)
    data_map = verify_webapp_init_data(payload.initData)

    # user_id вытаскиваем из initData (там user= urlencoded json)
    user_id = 0
    user_raw = data_map.get("user")
    if user_raw:
        import urllib.parse, json
        try:
            user_json = json.loads(urllib.parse.unquote(user_raw))
            user_id = int(user_json.get("id", 0))
        except Exception:
            user_id = 0
    if not user_id:
        raise HTTPException(401, detail="Cannot determine user_id from initData")

    items_total = sum(it.qty * it.price for it in payload.items)
    delivery_price, _km = calc_delivery_price(payload.lat, payload.lng)
    total = items_total + delivery_price

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
            address_text=payload.address_text,
            lat=payload.lat,
            lng=payload.lng,
            delivery_slot=payload.delivery_slot,
            items_total=items_total,
            delivery_price=delivery_price,
            total=total,
            status=OrderStatus.NEW.value,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.commit()

        keyboard = {
            "inline_keyboard": [[
                {"text": "✅ Принять заказ", "callback_data": f"accept:{order_id}"},
                {"text": "❌ Отклонить", "callback_data": f"reject:{order_id}"}
            ]]
        }

        await tg_call("sendMessage", {
            "chat_id": int(GROUP_CHAT_ID),
            "text": format_order_for_admin(order),
            "parse_mode": "Markdown",
            "reply_markup": keyboard,
        })

    return CreateOrderOut(order_id=order_id, total=total, delivery_price=delivery_price)


# =========================
# Cron job endpoint (Railway Cron -> call this)
# =========================
@app.post("/jobs/expire-payments")
async def expire_payments(x_job_secret: str | None = Header(default=None)):
    job_secret = os.getenv("JOB_SECRET", "")
    if job_secret and x_job_secret != job_secret:
        raise HTTPException(401, detail="Bad job secret")

    if not SessionLocal:
        raise HTTPException(500, detail="DB not configured")

    now = utcnow()
    async with SessionLocal() as session:
        q = select(Order).where(
            Order.status == OrderStatus.AWAITING_PAYMENT.value,
            Order.payment_deadline.is_not(None),
            Order.payment_deadline < now,
        )
        res = await session.execute(q)
        expired = res.scalars().all()
        if not expired:
            return {"ok": True, "expired": 0}

        for order in expired:
            order.status = OrderStatus.EXPIRED.value
            order.updated_at = now

        await session.commit()

        for order in expired:
            try:
                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": f"⛔️ Время оплаты по заказу `{order.id}` истекло. Заказ отменён.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            try:
                await tg_call("sendMessage", {
                    "chat_id": int(GROUP_CHAT_ID),
                    "text": f"⛔️ Заказ `{order.id}`: время оплаты истекло (15 минут).",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass

    return {"ok": True, "expired": len(expired)}


# =========================
# Telegram Webhook
# =========================
@app.post("/telegram/webhook")
async def telegram_webhook(req: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(401, detail="Bad webhook secret")

    update = await req.json()
    if not SessionLocal:
        return {"ok": True}

    # 1) Сообщения (команды в группе, чек от пользователя)
    msg = update.get("message")
    if msg:
        chat_id = msg.get("chat", {}).get("id")
        from_id = msg.get("from", {}).get("id")
        text = (msg.get("text") or "").strip()

        # 1.1 Админ задаёт реквизиты: /pay <текст>
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

            async with SessionLocal() as session:
                await set_setting(session, "payment_text", new_text)

            await tg_call("sendMessage", {"chat_id": chat_id, "text": "✅ Реквизиты сохранены в БД."})
            return {"ok": True}

        # 1.2 Пользователь прислал чек (photo/document)
        has_photo = bool(msg.get("photo"))
        has_doc = bool(msg.get("document"))
        if has_photo or has_doc:
            user_id = from_id
            file_id = None
            if has_photo:
                photo = msg["photo"][-1]  # самый большой
                file_id = photo.get("file_id")
            else:
                file_id = msg["document"].get("file_id")

            if file_id:
                async with SessionLocal() as session:
                    q = select(Order).where(
                        Order.user_id == user_id,
                        Order.status == OrderStatus.AWAITING_PAYMENT.value,
                    ).order_by(Order.created_at.desc()).limit(1)
                    res = await session.execute(q)
                    order = res.scalar_one_or_none()

                    if not order:
                        await tg_call("sendMessage", {"chat_id": user_id, "text": "Не нашёл активный заказ для чека."})
                        return {"ok": True}

                    order.receipt_file_id = file_id
                    order.receipt_message_id = msg.get("message_id")
                    order.status = OrderStatus.RECEIPT_SENT.value
                    order.updated_at = utcnow()
                    await session.commit()

                    keyboard = {
                        "inline_keyboard": [[
                            {"text": "✅ Подтвердить оплату", "callback_data": f"paid:{order.id}"},
                            {"text": "❌ Отклонить чек", "callback_data": f"reject_receipt:{order.id}"}
                        ]]
                    }
                    await tg_call("sendMessage", {
                        "chat_id": int(GROUP_CHAT_ID),
                        "text": f"📎 Чек по заказу `{order.id}` от пользователя `{user_id}`. Подтвердите оплату.",
                        "parse_mode": "Markdown",
                        "reply_markup": keyboard,
                    })

                    # переслать чек в группу
                    try:
                        if has_photo:
                            await tg_call("sendPhoto", {"chat_id": int(GROUP_CHAT_ID), "photo": file_id, "caption": f"Чек заказ `{order.id}`"})
                        else:
                            await tg_call("sendDocument", {"chat_id": int(GROUP_CHAT_ID), "document": file_id, "caption": f"Чек заказ `{order.id}`"})
                    except Exception:
                        pass

                    await tg_call("sendMessage", {
                        "chat_id": user_id,
                        "text": f"✅ Чек по заказу `{order.id}` получен. Ждём подтверждение.",
                        "parse_mode": "Markdown",
                    })
                    return {"ok": True}

        return {"ok": True}

    # 2) Callback-кнопки
    cb = update.get("callback_query")
    if not cb:
        return {"ok": True}

    cb_id = cb.get("id")
    data = (cb.get("data") or "").strip()
    from_user = cb.get("from", {})
    from_id = from_user.get("id")

    try:
        await tg_call("answerCallbackQuery", {"callback_query_id": cb_id})
    except Exception:
        pass

    is_admin = from_id in ADMIN_IDS

    async with SessionLocal() as session:
        if data.startswith("accept:"):
            if not is_admin:
                return {"ok": True}

            order_id = data.split("accept:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order or order.status != OrderStatus.NEW.value:
                return {"ok": True}

            payment_text = await get_setting(session, "payment_text", "")
            if not payment_text:
                await tg_call("sendMessage", {
                    "chat_id": int(GROUP_CHAT_ID),
                    "text": f"⚠️ Заказ `{order_id}` принят, но реквизиты не заданы.\nАдмин: /pay <текст>",
                    "parse_mode": "Markdown",
                })
                return {"ok": True}

            now = utcnow()
            order.status = OrderStatus.AWAITING_PAYMENT.value
            order.accepted_by = from_id
            order.accepted_at = now
            order.payment_deadline = now + timedelta(minutes=15)
            order.updated_at = now
            await session.commit()

            accepter = f"@{from_user.get('username')}" if from_user.get("username") else (from_user.get("first_name") or "Админ")

            await tg_call("sendMessage", {
                "chat_id": int(GROUP_CHAT_ID),
                "text": f"✅ Заказ `{order_id}` принят: {accepter}\nОтправил реквизиты пользователю. Таймер 15 минут запущен.",
                "parse_mode": "Markdown",
            })

            await tg_call("sendMessage", {
                "chat_id": order.user_id,
                "text": format_payment_to_user(order_id, payment_text),
                "parse_mode": "Markdown",
            })

            return {"ok": True}

        if data.startswith("reject:"):
            if not is_admin:
                return {"ok": True}
            order_id = data.split("reject:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}
            order.status = OrderStatus.CANCELLED.value
            order.updated_at = utcnow()
            await session.commit()

            try:
                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": f"❌ Заказ `{order_id}` отклонён.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            return {"ok": True}

        if data.startswith("paid:"):
            if not is_admin:
                return {"ok": True}
            order_id = data.split("paid:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}
            order.status = OrderStatus.PAID.value
            order.updated_at = utcnow()
            await session.commit()

            await tg_call("sendMessage", {
                "chat_id": int(GROUP_CHAT_ID),
                "text": f"✅ Оплата подтверждена по заказу `{order_id}`.",
                "parse_mode": "Markdown",
            })
            try:
                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": f"🎉 Оплата по заказу `{order_id}` подтверждена! Мы готовим доставку.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            return {"ok": True}

        if data.startswith("reject_receipt:"):
            if not is_admin:
                return {"ok": True}
            order_id = data.split("reject_receipt:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}
            # вернём в ожидание оплаты, если не истёк дедлайн
            now = utcnow()
            if order.payment_deadline and order.payment_deadline > now:
                order.status = OrderStatus.AWAITING_PAYMENT.value
            else:
                order.status = OrderStatus.EXPIRED.value
            order.receipt_file_id = None
            order.receipt_message_id = None
            order.updated_at = now
            await session.commit()

            try:
                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": f"❌ Чек по заказу `{order_id}` отклонён. Пришлите корректный чек.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            return {"ok": True}

    return {"ok": True}
