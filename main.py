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
GROUP_CHAT_ID = os.getenv("GROUP_CHAT_ID")  # Telegram group id (-100...)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "https://ret-ashy.vercel.app").split(",")


# =========================
# DB
# =========================
class Base(DeclarativeBase):
    pass


class OrderStatus(str, Enum):
    NEW = "NEW"                        # создан, отправлен в группу
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

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # short id
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    items: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    address_text: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    delivery_slot: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=OrderStatus.NEW.value)

    payment_method_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accepted_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

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
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(url, json=payload)
        data = r.json()
    if not data.get("ok"):
        raise HTTPException(500, detail=data)
    return data["result"]


def verify_webapp_init_data(init_data: str) -> dict[str, str]:
    """
    Проверка подписи Telegram WebApp initData.
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

    return data_map


def extract_user_id_from_init_data_map(data_map: dict[str, str]) -> int:
    user_raw = data_map.get("user")
    if not user_raw:
        return 0
    import urllib.parse, json
    try:
        user_json = json.loads(urllib.parse.unquote(user_raw))
        return int(user_json.get("id", 0))
    except Exception:
        return 0


def format_order_for_admin(order: Order) -> str:
    lines = []
    lines.append("🛒 *Новый заказ*")
    lines.append(f"🆔 Заказ: `{order.id}`")
    lines.append(f"👤 User ID: `{order.user_id}`")
    if order.address_text:
        lines.append(f"📍 Адрес: {order.address_text}")
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
    lines.append(f"*ИТОГО: {order.total} ₽*")
    return "\n".join(lines)


def format_payment_to_user(order_id: str, payment_title: str, payment_text: str) -> str:
    return (
        f"✅ Ваш заказ `{order_id}` принят.\n\n"
        f"💳 *Реквизиты для оплаты ({payment_title}):*\n{payment_text}\n\n"
        f"После оплаты отправьте сюда *фото/файл чека*."
    )


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
# Startup
# =========================
@app.on_event("startup")
async def _startup():
    if not engine:
        raise RuntimeError("DATABASE_URL not set")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # если реквизитов нет — создадим 2 примера (чтобы сразу работало)
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


@app.get("/health")
def health():
    return {"ok": True}


# =========================
# API (Mini App -> Backend)
# =========================
@app.post("/api/order", response_model=CreateOrderOut)
async def create_order(payload: CreateOrderIn):
    if not GROUP_CHAT_ID:
        raise HTTPException(500, detail="GROUP_CHAT_ID not set")
    if not SessionLocal:
        raise HTTPException(500, detail="DB not configured")
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
            address_text=payload.address_text,
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

        await tg_call("sendMessage", {
            "chat_id": int(GROUP_CHAT_ID),
            "text": format_order_for_admin(order),
            "parse_mode": "Markdown",
            "reply_markup": keyboard,
        })

    return CreateOrderOut(order_id=order_id, total=total)


# =========================
# Telegram Webhook
# =========================
@app.post("/telegram/webhook")
async def telegram_webhook(
    req: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(401, detail="Bad webhook secret")

    update = await req.json()
    if not SessionLocal:
        return {"ok": True}

    # -------------------------
    # Messages (commands, receipts)
    # -------------------------
    msg = update.get("message")
    if msg:
        chat_id = msg.get("chat", {}).get("id")
        from_id = msg.get("from", {}).get("id")
        text = (msg.get("text") or "").strip()

        # 1) Управление реквизитами (только в группе)
        # /paylist — список
        # /payadd <Название> | <Текст реквизитов>
        # /payoff <id> — выключить
        # /payon <id> — включить
        if GROUP_CHAT_ID and chat_id == int(GROUP_CHAT_ID) and text:
            if text.startswith("/paylist"):
                if from_id not in ADMIN_IDS:
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "⛔ Нет прав."})
                    return {"ok": True}
                async with SessionLocal() as session:
                    res = await session.execute(select(PaymentMethod).order_by(PaymentMethod.id.asc()))
                    rows = res.scalars().all()
                if not rows:
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "Реквизитов нет."})
                else:
                    lines = ["💳 *Реквизиты:*"]
                    for r in rows:
                        state = "✅" if r.is_active else "⛔"
                        lines.append(f"{state} `{r.id}` — *{r.title}*")
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "Markdown"})
                return {"ok": True}

            if text.startswith("/payadd"):
                if from_id not in ADMIN_IDS:
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "⛔ Нет прав."})
                    return {"ok": True}
                payload_txt = text[len("/payadd"):].strip()
                if "|" not in payload_txt:
                    await tg_call("sendMessage", {
                        "chat_id": chat_id,
                        "text": "Формат:\n/payadd Название | Текст реквизитов\n\nПример:\n/payadd Карта Тинькофф | Карта: ....",
                    })
                    return {"ok": True}
                title, ptext = [x.strip() for x in payload_txt.split("|", 1)]
                if not title or not ptext:
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "Название и текст не должны быть пустыми."})
                    return {"ok": True}

                async with SessionLocal() as session:
                    session.add(PaymentMethod(title=title, text=ptext, is_active=True, created_at=utcnow()))
                    await session.commit()

                await tg_call("sendMessage", {"chat_id": chat_id, "text": "✅ Реквизит добавлен."})
                return {"ok": True}

            if text.startswith("/payoff") or text.startswith("/payon"):
                if from_id not in ADMIN_IDS:
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "⛔ Нет прав."})
                    return {"ok": True}
                parts = text.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    await tg_call("sendMessage", {"chat_id": chat_id, "text": "Формат: /payoff <id> или /payon <id>"})
                    return {"ok": True}
                mid = int(parts[1])
                new_state = text.startswith("/payon")
                async with SessionLocal() as session:
                    m = await session.get(PaymentMethod, mid)
                    if not m:
                        await tg_call("sendMessage", {"chat_id": chat_id, "text": "Не найдено."})
                        return {"ok": True}
                    m.is_active = new_state
                    await session.commit()
                await tg_call("sendMessage", {"chat_id": chat_id, "text": f"✅ Обновлено: {mid} → {'active' if new_state else 'inactive'}"})
                return {"ok": True}

        # 2) Пользователь прислал чек (photo/document)
        has_photo = bool(msg.get("photo"))
        has_doc = bool(msg.get("document"))
        if has_photo or has_doc:
            user_id = from_id
            file_id = None
            if has_photo:
                photo = msg["photo"][-1]
                file_id = photo.get("file_id")
            else:
                file_id = msg["document"].get("file_id")

            if file_id:
                async with SessionLocal() as session:
                    q = (
                        select(Order)
                        .where(
                            Order.user_id == user_id,
                            Order.status == OrderStatus.AWAITING_PAYMENT.value,
                        )
                        .order_by(Order.created_at.desc())
                        .limit(1)
                    )
                    res = await session.execute(q)
                    order = res.scalar_one_or_none()

                    if not order:
                        await tg_call("sendMessage", {"chat_id": user_id, "text": "Не нашёл заказ, который ожидает оплату."})
                        return {"ok": True}

                    order.receipt_file_id = file_id
                    order.receipt_message_id = msg.get("message_id")
                    order.status = OrderStatus.RECEIPT_SENT.value
                    order.updated_at = utcnow()
                    await session.commit()

                    keyboard = {
                        "inline_keyboard": [[
                            {"text": "✅ Подтвердить оплату", "callback_data": f"paid:{order.id}"},
                            {"text": "❌ Отклонить чек", "callback_data": f"reject_receipt:{order.id}"},
                        ]]
                    }

                    await tg_call("sendMessage", {
                        "chat_id": int(GROUP_CHAT_ID),
                        "text": f"📎 Чек по заказу `{order.id}` от пользователя `{user_id}`. Проверьте и подтвердите оплату.",
                        "parse_mode": "Markdown",
                        "reply_markup": keyboard,
                    })

                    # Переслать чек в группу
                    try:
                        if has_photo:
                            await tg_call("sendPhoto", {"chat_id": int(GROUP_CHAT_ID), "photo": file_id, "caption": f"Чек заказ `{order.id}`"})
                        else:
                            await tg_call("sendDocument", {"chat_id": int(GROUP_CHAT_ID), "document": file_id, "caption": f"Чек заказ `{order.id}`"})
                    except Exception:
                        pass

                    await tg_call("sendMessage", {
                        "chat_id": user_id,
                        "text": f"✅ Чек по заказу `{order.id}` получен. Ожидайте подтверждение.",
                        "parse_mode": "Markdown",
                    })
                    return {"ok": True}

        return {"ok": True}

    # -------------------------
    # Callback buttons
    # -------------------------
    cb = update.get("callback_query")
    if not cb:
        return {"ok": True}

    cb_id = cb.get("id")
    data = (cb.get("data") or "").strip()
    from_user = cb.get("from", {})
    from_id = from_user.get("id")
    is_admin = from_id in ADMIN_IDS

    try:
        await tg_call("answerCallbackQuery", {"callback_query_id": cb_id})
    except Exception:
        pass

    async with SessionLocal() as session:
        # 1) "Выбрать реквизиты" -> показать список кнопок
        if data.startswith("choosepay:"):
            if not is_admin:
                return {"ok": True}

            order_id = data.split("choosepay:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order or order.status != OrderStatus.NEW.value:
                return {"ok": True}

            res = await session.execute(
                select(PaymentMethod).where(PaymentMethod.is_active == True).order_by(PaymentMethod.id.asc())
            )
            methods = res.scalars().all()
            if not methods:
                await tg_call("sendMessage", {
                    "chat_id": int(GROUP_CHAT_ID),
                    "text": "⚠️ Нет активных реквизитов. Добавь: /payadd Название | Текст",
                })
                return {"ok": True}

            buttons = []
            row = []
            for m in methods:
                row.append({"text": m.title, "callback_data": f"setpay:{order_id}:{m.id}"})
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)

            await tg_call("sendMessage", {
                "chat_id": int(GROUP_CHAT_ID),
                "text": f"Выберите реквизит для заказа `{order_id}`:",
                "parse_mode": "Markdown",
                "reply_markup": {"inline_keyboard": buttons},
            })
            return {"ok": True}

        # 2) Установить реквизит и отправить пользователю
        if data.startswith("setpay:"):
            if not is_admin:
                return {"ok": True}
            # setpay:<order_id>:<method_id>
            try:
                _, order_id, mid = data.split(":", 2)
                method_id = int(mid)
            except Exception:
                return {"ok": True}

            order = await session.get(Order, order_id)
            if not order or order.status != OrderStatus.NEW.value:
                return {"ok": True}

            method = await session.get(PaymentMethod, method_id)
            if not method or not method.is_active:
                await tg_call("sendMessage", {"chat_id": int(GROUP_CHAT_ID), "text": "⚠️ Реквизит не найден/неактивен."})
                return {"ok": True}

            now = utcnow()
            order.payment_method_id = method.id
            order.accepted_by = from_id
            order.accepted_at = now
            order.status = OrderStatus.AWAITING_PAYMENT.value
            order.updated_at = now
            await session.commit()

            accepter = f"@{from_user.get('username')}" if from_user.get("username") else (from_user.get("first_name") or "Админ")

            await tg_call("sendMessage", {
                "chat_id": int(GROUP_CHAT_ID),
                "text": f"✅ Заказ `{order_id}`: выбран реквизит *{method.title}* (админ {accepter}). Реквизиты отправлены пользователю.",
                "parse_mode": "Markdown",
            })

            await tg_call("sendMessage", {
                "chat_id": order.user_id,
                "text": format_payment_to_user(order_id, method.title, method.text),
                "parse_mode": "Markdown",
            })
            return {"ok": True}

        # 3) Отмена заказа
        if data.startswith("cancel:"):
            if not is_admin:
                return {"ok": True}
            order_id = data.split("cancel:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}
            order.status = OrderStatus.CANCELLED.value
            order.updated_at = utcnow()
            await session.commit()
            try:
                await tg_call("sendMessage", {"chat_id": order.user_id, "text": f"❌ Заказ `{order_id}` отменён.", "parse_mode": "Markdown"})
            except Exception:
                pass
            await tg_call("sendMessage", {"chat_id": int(GROUP_CHAT_ID), "text": f"❌ Заказ `{order_id}` отменён.", "parse_mode": "Markdown"})
            return {"ok": True}

        # 4) Подтвердить оплату
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
                    "text": f"🎉 Оплата по заказу `{order_id}` подтверждена! Мы приступили к выполнению.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            return {"ok": True}

        # 5) Отклонить чек
        if data.startswith("reject_receipt:"):
            if not is_admin:
                return {"ok": True}
            order_id = data.split("reject_receipt:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}

            order.status = OrderStatus.AWAITING_PAYMENT.value
            order.receipt_file_id = None
            order.receipt_message_id = None
            order.updated_at = utcnow()
            await session.commit()

            try:
                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": f"❌ Чек по заказу `{order_id}` отклонён. Пожалуйста, отправьте корректный чек.",
                    "parse_mode": "Markdown",
                })
            except Exception:
                pass
            await tg_call("sendMessage", {"chat_id": int(GROUP_CHAT_ID), "text": f"❌ Чек отклонён по заказу `{order_id}`.", "parse_mode": "Markdown"})
            return {"ok": True}

    return {"ok": True}
