import os
import uuid
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional

import httpx
import phonenumbers
from fastapi import FastAPI, Header, HTTPException, Request, Query
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
    text as sql_text,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").strip().lstrip("@")  # <-- добавь в Railway Variables

GROUP_CHAT_ID_RAW = os.getenv("GROUP_CHAT_ID")  # -100...
GROUP_CHAT_ID: Optional[int] = None
if GROUP_CHAT_ID_RAW:
    try:
        GROUP_CHAT_ID = int(GROUP_CHAT_ID_RAW)
    except Exception:
        GROUP_CHAT_ID = None

MINIAPP_URL = os.getenv("MINIAPP_URL", "https://ret-ashy.vercel.app")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
REBUILD_SECRET = os.getenv("REBUILD_SECRET", "")

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "")
ADMIN_IDS = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Referral/Loyalty rules
WITHDRAW_MIN_RUB = 5000  # minimum to request payout
REFERRAL_MAX_PERCENT = 20
REFERRAL_TRIGGER_TOTAL = 5000  # referral must spend >= this to count towards levels and payouts
REFERRAL_PAYOUT_ORDERS_LIMIT = 3  # only first N paid orders of referral give payout
LOYALTY_MAX_REDEEM_PERCENT = 20

REG_BONUS_POINTS = 2000
REG_BONUS_WINDOW_HOURS = 24


# =========================
# DB Models
# =========================
class Base(DeclarativeBase):
    pass


class OrderStatus(str, Enum):
    NEW = "NEW"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    RECEIPT_UPLOADED = "RECEIPT_UPLOADED"
    UNDER_REVIEW = "UNDER_REVIEW"
    PAID = "PAID"
    REJECTED = "REJECTED"
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
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    items: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default=OrderStatus.NEW.value)

    payment_method_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    accepted_by: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    receipt_file_id: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    receipt_kind: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # photo/document
    receipt_message_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# -------- Users / Registration
class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    birth_date: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # YYYY-MM-DD

    is_registered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # registration bonus control
    reg_bonus_deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    reg_bonus_granted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


# -------- Consents (rules accept/decline with cooldown)
class ConsentKind(str, Enum):
    REFERRAL = "REFERRAL"
    LOYALTY = "LOYALTY"


class Consent(Base):
    __tablename__ = "consents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # ConsentKind
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # None = not decided
    cooldown_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# -------- Wallets + ledger
class WalletKind(str, Enum):
    LOYALTY = "LOYALTY"      # points
    REFERRAL = "REFERRAL"    # rub/points for withdrawal


class Wallet(Base):
    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # WalletKind
    balance: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LedgerType(str, Enum):
    CREDIT = "CREDIT"
    DEBIT = "DEBIT"
    EXCHANGE = "EXCHANGE"


class WalletLedger(Base):
    __tablename__ = "wallet_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    wallet_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # WalletKind
    entry_type: Mapped[str] = mapped_column(String(16), nullable=False)  # LedgerType
    amount: Mapped[int] = mapped_column(Integer, nullable=False)  # positive number
    meta: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# -------- Referral codes/links and relations
class ReferralCode(Base):
    __tablename__ = "referral_codes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ReferralLink(Base):
    __tablename__ = "referral_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    token: Mapped[str] = mapped_column(String(48), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ReferralRelation(Base):
    __tablename__ = "referral_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    referrer_user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    referral_user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False, unique=True)

    # binding source
    bound_by: Mapped[str] = mapped_column(String(16), nullable=False)  # LINK|CODE
    bound_value: Mapped[str] = mapped_column(String(64), nullable=False)  # token/code

    # progress tracking
    referral_paid_orders: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    referral_paid_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # payouts
    payouts_done_orders: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # how many paid orders generated payout
    is_detached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# -------- Withdrawal
class WithdrawStatus(str, Enum):
    NEW = "NEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PAID = "PAID"


class WithdrawRequest(Base):
    __tablename__ = "withdraw_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    sbp_phone: Mapped[str] = mapped_column(String(32), nullable=False)
    bank_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=WithdrawStatus.NEW.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


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

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# In-memory states (simple FSM)
# =========================
class RegStep(str, Enum):
    NONE = "NONE"
    NAME = "NAME"
    PHONE = "PHONE"
    BIRTH = "BIRTH"


REG_STATE: dict[int, str] = {}  # user_id -> state
REG_TEMP: dict[int, dict[str, str]] = {}  # user_id -> collected fields


# =========================
# Utils
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def require_db():
    if not SessionLocal or not DB_READY:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {DB_ERROR or 'not ready'}")


def is_admin(user_id: Optional[int]) -> bool:
    return bool(user_id and user_id in ADMIN_IDS)


def normalize_phone(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        p = phonenumbers.parse(raw, "RU")
        if not phonenumbers.is_valid_number(p):
            return None
        return phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        return None


def parse_birth_date(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    try:
        parts = raw.split("-")
        if len(parts) != 3:
            return None
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if y < 1900 or y > 2100:
            return None
        if m < 1 or m > 12:
            return None
        if d < 1 or d > 31:
            return None
        return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        return None


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


def format_payment_to_user(order_id: str, payment_title: str, payment_text: str) -> str:
    return (
        f"✅ Ваш заказ `{order_id}` принят.\n\n"
        f"💳 *Реквизиты для оплаты ({payment_title}):*\n{payment_text}\n\n"
        f"1) Оплатите\n"
        f"2) Отправьте сюда *фото/файл чека*\n"
        f"3) Затем нажмите кнопку *✅ Оплатил*"
    )


def main_menu_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [{"text": "🛍 Каталог", "web_app": {"url": MINIAPP_URL}}],
            [{"text": "🎁 Лояльность", "callback_data": "loyalty:open"}],
            [{"text": "🤝 Рефералка", "callback_data": "ref:open"}],
            [{"text": "👤 Профиль", "callback_data": "profile:open"}],
        ]
    }


def rules_referral_text() -> str:
    return (
        "🤝 *Реферальная программа — правила*\n\n"
        "1) Максимальный процент начислений: *20%*.\n"
        "2) Уровни:\n"
        "   • *5%* — когда 3 реферала сделали ≥1 заказ и суммарно ≥ 5000 ₽\n"
        "   • *10%* — до 10 рефералов\n"
        "   • *15%* — до 20 рефералов\n"
        "   • *20%* — если 20 рефералов\n"
        "3) Реферал закрепляется по ссылке или промокоду.\n"
        "4) Чтобы реферал учитывался для уровней и выплат: он должен оплатить заказов суммарно минимум *5000 ₽*.\n"
        "5) Начисления идут только за *первые 3 оплаченных заказа* реферала.\n"
        "   После 3-го заказа реферал отвязывается (дальше начислений нет), но достигнутый уровень сохраняется.\n"
        f"6) Вывод доступен от *{WITHDRAW_MIN_RUB} ₽* на СБП.\n"
        "7) Обмен баллов (в обе стороны): *2 к 1*.\n\n"
        "Нажимая «Принять», вы соглашаетесь с правилами."
    )


def rules_loyalty_text() -> str:
    return (
        "🎁 *Программа лояльности — правила*\n\n"
        "1) Баллы начисляются после подтверждения оплаты заказа.\n"
        "2) Уровни кешбэка:\n"
        "   • 3% — базовый\n"
        "   • 5% — если потрачено ≥ 25 000 ₽\n"
        "   • 7% — если потрачено ≥ 70 000 ₽\n"
        "   • 10% — если потрачено ≥ 100 000 ₽\n"
        f"3) Списывать можно до *{LOYALTY_MAX_REDEEM_PERCENT}%* суммы заказа (в следующем шаге подключим к заказу).\n"
        "4) Обмен баллов (в обе стороны): *2 к 1*.\n\n"
        "Нажимая «Принять», вы соглашаетесь с правилами."
    )


# =========================
# Wallet helpers
# =========================
async def get_or_create_wallet(session: AsyncSession, user_id: int, kind: str):
    q = select(Wallet).where(Wallet.user_id == user_id, Wallet.kind == kind)
    res = await session.execute(q)
    w = res.scalar_one_or_none()
    if w:
        return w
    now = utcnow()
    w = Wallet(user_id=user_id, kind=kind, balance=0, created_at=now, updated_at=now)
    session.add(w)
    await session.flush()
    return w


async def ledger_add(session: AsyncSession, user_id: int, kind: str, entry_type: str, amount: int, meta: dict):
    session.add(
        WalletLedger(
            user_id=user_id,
            wallet_kind=kind,
            entry_type=entry_type,
            amount=amount,
            meta=meta or {},
            created_at=utcnow(),
        )
    )


async def wallet_credit(session: AsyncSession, user_id: int, kind: str, amount: int, meta: dict):
    if amount <= 0:
        return
    w = await get_or_create_wallet(session, user_id, kind)
    w.balance += amount
    w.updated_at = utcnow()
    await ledger_add(session, user_id, kind, LedgerType.CREDIT.value, amount, meta)


async def wallet_debit(session: AsyncSession, user_id: int, kind: str, amount: int, meta: dict) -> bool:
    if amount <= 0:
        return True
    w = await get_or_create_wallet(session, user_id, kind)
    if w.balance < amount:
        return False
    w.balance -= amount
    w.updated_at = utcnow()
    await ledger_add(session, user_id, kind, LedgerType.DEBIT.value, amount, meta)
    return True


# =========================
# Referral percent and levels
# =========================
async def count_qualified_referrals(session: AsyncSession, referrer_id: int) -> int:
    q = select(ReferralRelation).where(ReferralRelation.referrer_user_id == referrer_id)
    res = await session.execute(q)
    rels = res.scalars().all()
    n = 0
    for r in rels:
        if r.referral_paid_orders >= 1 and r.referral_paid_total >= REFERRAL_TRIGGER_TOTAL:
            n += 1
    return n


def referral_percent_for_count(qty: int) -> int:
    if qty >= 20:
        return 20
    if qty >= 11:
        return 15
    if qty >= 4:
        return 10
    if qty >= 3:
        return 5
    return 0


def loyalty_percent_for_spend(total_spend: int) -> int:
    if total_spend >= 100_000:
        return 10
    if total_spend >= 70_000:
        return 7
    if total_spend >= 25_000:
        return 5
    return 3


# =========================
# Consents helpers
# =========================
async def get_consent(session: AsyncSession, user_id: int, kind: str) -> Consent:
    q = select(Consent).where(Consent.user_id == user_id, Consent.kind == kind)
    res = await session.execute(q)
    c = res.scalar_one_or_none()
    if c:
        return c
    now = utcnow()
    c = Consent(
        user_id=user_id,
        kind=kind,
        accepted=None,
        cooldown_until=None,
        created_at=now,
        updated_at=now,
    )
    session.add(c)
    await session.flush()
    return c


def consent_can_show(c: Consent) -> bool:
    if c.accepted is True:
        return True
    if c.accepted is False:
        if c.cooldown_until and utcnow() < c.cooldown_until:
            return False
        return True
    return True


# =========================
# Referral binding: link or code
# =========================
def build_referral_start_payload(token: str) -> str:
    return f"ref_{token}"


def build_referral_tg_link(token: str) -> str:
    payload = build_referral_start_payload(token)
    if BOT_USERNAME:
        return f"https://t.me/{BOT_USERNAME}?start={payload}"
    return f"t.me/<BOT_USERNAME>?start={payload}"


async def ensure_referral_assets(session: AsyncSession, user_id: int):
    q1 = select(ReferralCode).where(ReferralCode.owner_user_id == user_id, ReferralCode.is_active == True)  # noqa: E712
    r1 = await session.execute(q1)
    code = r1.scalar_one_or_none()
    if not code:
        while True:
            new_code = f"ROSE{uuid.uuid4().hex[:6].upper()}"
            qx = select(ReferralCode).where(ReferralCode.code == new_code)
            rx = await session.execute(qx)
            if not rx.scalar_one_or_none():
                break
        code = ReferralCode(owner_user_id=user_id, code=new_code, is_active=True, created_at=utcnow())
        session.add(code)

    q2 = select(ReferralLink).where(ReferralLink.owner_user_id == user_id, ReferralLink.is_active == True)  # noqa: E712
    r2 = await session.execute(q2)
    link = r2.scalar_one_or_none()
    if not link:
        while True:
            token = uuid.uuid4().hex + uuid.uuid4().hex[:6]
            qx = select(ReferralLink).where(ReferralLink.token == token)
            rx = await session.execute(qx)
            if not rx.scalar_one_or_none():
                break
        link = ReferralLink(owner_user_id=user_id, token=token, is_active=True, created_at=utcnow())
        session.add(link)

    await session.commit()
    return code, link


async def bind_referral(session: AsyncSession, referrer_id: int, referral_id: int, bound_by: str, bound_value: str) -> bool:
    if referrer_id == referral_id:
        return False
    q = select(ReferralRelation).where(ReferralRelation.referral_user_id == referral_id)
    res = await session.execute(q)
    existing = res.scalar_one_or_none()
    if existing:
        return False

    now = utcnow()
    rel = ReferralRelation(
        referrer_user_id=referrer_id,
        referral_user_id=referral_id,
        bound_by=bound_by,
        bound_value=bound_value,
        referral_paid_orders=0,
        referral_paid_total=0,
        payouts_done_orders=0,
        is_detached=False,
        created_at=now,
        updated_at=now,
    )
    session.add(rel)
    await session.commit()
    return True


# =========================
# Schema helpers (ONLY from rebuild endpoint)
# =========================
async def ensure_schema(conn):
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS payment_method_id INTEGER;"))
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS accepted_by BIGINT;"))
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ;"))
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS receipt_file_id TEXT;"))
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS receipt_kind VARCHAR(16);"))
    await conn.execute(sql_text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS receipt_message_id BIGINT;"))

    await conn.execute(sql_text("ALTER TABLE payment_methods ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;"))
    await conn.execute(sql_text("ALTER TABLE payment_methods ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();"))
    await conn.execute(sql_text("UPDATE payment_methods SET created_at = now() WHERE created_at IS NULL;"))
    await conn.execute(sql_text("ALTER TABLE payment_methods ALTER COLUMN created_at SET NOT NULL;"))


async def rebuild_db(conn):
    await conn.execute(sql_text("DROP TABLE IF EXISTS withdraw_requests CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS wallet_ledger CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS wallets CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS referral_relations CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS referral_links CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS referral_codes CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS consents CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS users CASCADE;"))

    await conn.execute(sql_text("DROP TABLE IF EXISTS orders CASCADE;"))
    await conn.execute(sql_text("DROP TABLE IF EXISTS payment_methods CASCADE;"))

    await conn.run_sync(Base.metadata.create_all)
    await ensure_schema(conn)


async def seed_methods(session: AsyncSession):
    res = await session.execute(select(PaymentMethod).order_by(PaymentMethod.id.asc()))
    methods = res.scalars().all()
    if not methods:
        now = utcnow()
        session.add_all(
            [
                PaymentMethod(title="Карта 1", text="Карта: 0000 0000 0000 0000\nПолучатель: ...", is_active=True, created_at=now),
                PaymentMethod(title="СБП", text="СБП по номеру: +7...\nБанк: ...\nПолучатель: ...", is_active=True, created_at=now),
            ]
        )
        await session.commit()


# =========================
# Startup (NO schema changes)
# =========================
@app.on_event("startup")
async def _startup():
    global DB_READY, DB_ERROR

    if not engine or not SessionLocal:
        DB_READY = False
        DB_ERROR = "DATABASE_URL not set"
        return

    try:
        async with engine.connect() as conn:
            await conn.execute(sql_text("SELECT 1;"))
        DB_READY = True
        DB_ERROR = ""
    except Exception as e:
        DB_READY = False
        DB_ERROR = repr(e)


@app.get("/health")
def health():
    return {"ok": True, "db_ready": DB_READY, "db_error": DB_ERROR}


# =========================
# Admin rebuild endpoint (runtime)
# =========================
@app.get("/admin/rebuild-db")
async def admin_rebuild_db_get(secret: str = Query(default="")):
    require_db()
    if not REBUILD_SECRET or secret != REBUILD_SECRET:
        raise HTTPException(401, detail="Bad secret")

    async with engine.begin() as conn:
        await rebuild_db(conn)

    async with SessionLocal() as session:
        await seed_methods(session)

    return {"ok": True, "rebuild": True}


@app.post("/admin/rebuild-db")
async def admin_rebuild_db_post(secret: str = Query(default="")):
    return await admin_rebuild_db_get(secret=secret)


# =========================
# Miniapp API
# =========================
class OrderItemIn(BaseModel):
    product_id: int
    title: str
    price: int
    qty: int = Field(ge=1)


class CreateOrderIn(BaseModel):
    initData: str
    items: list[OrderItemIn]


class CreateOrderOut(BaseModel):
    ok: bool = True
    order_id: str
    total: int


@app.post("/api/order", response_model=CreateOrderOut)
async def create_order(payload: CreateOrderIn):
    require_db()
    if not GROUP_CHAT_ID:
        raise HTTPException(500, detail="GROUP_CHAT_ID not set or invalid")
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
        # ensure user row exists (важно для регистрационных сценариев)
        u = await session.get(User, user_id)
        if not u:
            u = User(
                user_id=user_id,
                full_name=None,
                phone=None,
                birth_date=None,
                is_registered=False,
                created_at=now,
                updated_at=now,
                reg_bonus_deadline_at=None,
                reg_bonus_granted=False,
            )
            session.add(u)
            await session.flush()

        order = Order(
            id=order_id,
            user_id=user_id,
            items=items_struct,
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
        "chat_id": GROUP_CHAT_ID,
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

    require_db()
    update = await req.json()

    msg = update.get("message")
    if msg:
        chat_id = msg.get("chat", {}).get("id")
        from_id = msg.get("from", {}).get("id")
        text = (msg.get("text") or "").strip()

        # ====== /start and deep links ======
        if text.startswith("/start"):
            async with SessionLocal() as session:
                u = await session.get(User, from_id)
                if not u:
                    now = utcnow()
                    u = User(
                        user_id=from_id,
                        full_name=None,
                        phone=None,
                        birth_date=None,
                        is_registered=False,
                        created_at=now,
                        updated_at=now,
                        reg_bonus_deadline_at=None,
                        reg_bonus_granted=False,
                    )
                    session.add(u)
                    await session.commit()

            parts = text.split(maxsplit=1)
            if len(parts) == 2:
                payload = parts[1].strip()
                if payload.startswith("ref_"):
                    token = payload.replace("ref_", "", 1).strip()
                    async with SessionLocal() as session:
                        q = select(ReferralLink).where(ReferralLink.token == token, ReferralLink.is_active == True)  # noqa: E712
                        res = await session.execute(q)
                        link = res.scalar_one_or_none()
                        if link and link.owner_user_id != from_id:
                            ok = await bind_referral(session, link.owner_user_id, from_id, "LINK", token)
                            if ok:
                                await tg_call("sendMessage", {"chat_id": from_id, "text": "✅ Реферал закреплён. Добро пожаловать!"})
                            else:
                                await tg_call("sendMessage", {"chat_id": from_id, "text": "ℹ️ Рефералка уже была закреплена ранее."})

            await tg_call("sendMessage", {
                "chat_id": chat_id,
                "text": "Добро пожаловать в 🌹 Магазин роз!\nВыберите действие:",
                "reply_markup": main_menu_keyboard(),
            })
            return {"ok": True}

        # ====== message states (registration + exchange + withdrawal) ======
        if chat_id == from_id and REG_STATE.get(from_id):
            state = REG_STATE.get(from_id)

            # ---- Registration ----
            if state == RegStep.NAME.value:
                name = (text or "").strip()
                if len(name) < 2:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите имя и фамилию (минимум 2 символа)."})
                    return {"ok": True}
                REG_TEMP[from_id] = {"full_name": name}
                REG_STATE[from_id] = RegStep.PHONE.value
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите номер телефона (можно с +7...)."})
                return {"ok": True}

            if state == RegStep.PHONE.value:
                phone = normalize_phone(text)
                if not phone:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Телефон не распознан. Пример: +79991234567"})
                    return {"ok": True}
                REG_TEMP.setdefault(from_id, {})["phone"] = phone
                REG_STATE[from_id] = RegStep.BIRTH.value
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите дату рождения в формате YYYY-MM-DD (например 1998-05-21)."})
                return {"ok": True}

            if state == RegStep.BIRTH.value:
                bd = parse_birth_date(text)
                if not bd:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Неверный формат. Введите дату рождения как YYYY-MM-DD."})
                    return {"ok": True}

                data = REG_TEMP.get(from_id, {})
                full_name = data.get("full_name")
                phone = data.get("phone")

                async with SessionLocal() as session:
                    u = await session.get(User, from_id)
                    if not u:
                        now = utcnow()
                        u = User(
                            user_id=from_id,
                            full_name=None,
                            phone=None,
                            birth_date=None,
                            is_registered=False,
                            created_at=now,
                            updated_at=now,
                            reg_bonus_deadline_at=None,
                            reg_bonus_granted=False,
                        )
                        session.add(u)
                        await session.flush()

                    u.full_name = full_name
                    u.phone = phone
                    u.birth_date = bd
                    u.is_registered = True
                    u.updated_at = utcnow()

                    # Бонус: только если уложился в окно 24 часа
                    if u.reg_bonus_deadline_at and (utcnow() <= u.reg_bonus_deadline_at) and (not u.reg_bonus_granted):
                        await wallet_credit(session, from_id, WalletKind.LOYALTY.value, REG_BONUS_POINTS, {"reason": "reg_bonus"})
                        u.reg_bonus_granted = True

                    await session.commit()

                REG_STATE.pop(from_id, None)
                REG_TEMP.pop(from_id, None)

                await tg_call("sendMessage", {"chat_id": from_id, "text": "✅ Регистрация завершена. Теперь доступны Лояльность и Рефералка."})
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Меню:", "reply_markup": main_menu_keyboard()})
                return {"ok": True}

            # ---- Exchange Loyalty -> Referral ----
            if state == "EX_L2R":
                raw = (text or "").strip()
                if not raw.isdigit():
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите число (например 200)."})
                    return {"ok": True}
                amount = int(raw)
                if amount < 2 or amount % 2 != 0:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите *чётное* число ≥ 2. Курс 2:1 (например 100 → 50).", "parse_mode": "Markdown"})
                    return {"ok": True}

                credit = amount // 2

                async with SessionLocal() as session:
                    ok = await wallet_debit(session, from_id, WalletKind.LOYALTY.value, amount, {"reason": "exchange_loyalty_to_referral"})
                    if not ok:
                        w = await get_or_create_wallet(session, from_id, WalletKind.LOYALTY.value)
                        await session.commit()
                        await tg_call("sendMessage", {"chat_id": from_id, "text": f"❌ Недостаточно баллов лояльности. Сейчас: {w.balance}."})
                        return {"ok": True}

                    await wallet_credit(session, from_id, WalletKind.REFERRAL.value, credit, {"reason": "exchange_from_loyalty", "spent": amount})
                    await session.commit()

                    wL = await get_or_create_wallet(session, from_id, WalletKind.LOYALTY.value)
                    wR = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)
                    await session.commit()

                REG_STATE.pop(from_id, None)
                await tg_call("sendMessage", {"chat_id": from_id, "text": f"✅ Обмен: -{amount} Лояльность → +{credit} Рефералка.\n\nБаланс:\n🎁 Лояльность: {wL.balance}\n🤝 Рефералка: {wR.balance}"})
                return {"ok": True}

            # ---- Exchange Referral -> Loyalty ----
            if state == "EX_R2L":
                raw = (text or "").strip()
                if not raw.isdigit():
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите число (например 200)."})
                    return {"ok": True}
                amount = int(raw)
                if amount < 2 or amount % 2 != 0:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите *чётное* число ≥ 2. Курс 2:1 (например 100 → 50).", "parse_mode": "Markdown"})
                    return {"ok": True}

                credit = amount // 2

                async with SessionLocal() as session:
                    ok = await wallet_debit(session, from_id, WalletKind.REFERRAL.value, amount, {"reason": "exchange_referral_to_loyalty"})
                    if not ok:
                        w = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)
                        await session.commit()
                        await tg_call("sendMessage", {"chat_id": from_id, "text": f"❌ Недостаточно реферальных баллов. Сейчас: {w.balance}."})
                        return {"ok": True}

                    await wallet_credit(session, from_id, WalletKind.LOYALTY.value, credit, {"reason": "exchange_from_referral", "spent": amount})
                    await session.commit()

                    wL = await get_or_create_wallet(session, from_id, WalletKind.LOYALTY.value)
                    wR = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)
                    await session.commit()

                REG_STATE.pop(from_id, None)
                await tg_call("sendMessage", {"chat_id": from_id, "text": f"✅ Обмен: -{amount} Рефералка → +{credit} Лояльность.\n\nБаланс:\n🎁 Лояльность: {wL.balance}\n🤝 Рефералка: {wR.balance}"})
                return {"ok": True}

            # ---- Withdrawal flow ----
            if state == "WD_AMOUNT":
                raw = (text or "").strip()
                if not raw.isdigit():
                    await tg_call("sendMessage", {"chat_id": from_id, "text": f"Введите сумму числом (минимум {WITHDRAW_MIN_RUB})."})
                    return {"ok": True}
                amount = int(raw)
                if amount < WITHDRAW_MIN_RUB:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": f"Минимум для вывода: {WITHDRAW_MIN_RUB}. Введите другую сумму."})
                    return {"ok": True}

                REG_TEMP[from_id] = {"wd_amount": str(amount)}
                REG_STATE[from_id] = "WD_PHONE"
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите номер телефона для СБП (например +79991234567)."})
                return {"ok": True}

            if state == "WD_PHONE":
                phone = normalize_phone(text)
                if not phone:
                    await tg_call("sendMessage", {"chat_id": from_id, "text": "Телефон не распознан. Пример: +79991234567"})
                    return {"ok": True}
                REG_TEMP.setdefault(from_id, {})["wd_phone"] = phone
                REG_STATE[from_id] = "WD_BANK"
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите банк (можно коротко), или напишите '-' если не хотите указывать."})
                return {"ok": True}

            if state == "WD_BANK":
                bank = (text or "").strip()
                if bank == "-":
                    bank = ""

                amount = int(REG_TEMP.get(from_id, {}).get("wd_amount", "0") or "0")
                phone = REG_TEMP.get(from_id, {}).get("wd_phone", "")

                async with SessionLocal() as session:
                    wR = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)
                    if wR.balance < amount:
                        await session.commit()
                        REG_STATE.pop(from_id, None)
                        REG_TEMP.pop(from_id, None)
                        await tg_call("sendMessage", {"chat_id": from_id, "text": f"❌ Недостаточно реферальных баллов. Сейчас: {wR.balance}."})
                        return {"ok": True}

                    session.add(WithdrawRequest(
                        user_id=from_id,
                        amount=amount,
                        sbp_phone=phone,
                        bank_name=(bank or None),
                        status=WithdrawStatus.NEW.value,
                        created_at=utcnow(),
                    ))
                    await session.commit()

                REG_STATE.pop(from_id, None)
                REG_TEMP.pop(from_id, None)
                await tg_call("sendMessage", {"chat_id": from_id, "text": f"✅ Заявка на вывод создана: {amount} ₽ на СБП {phone}. Админ рассмотрит."})
                return {"ok": True}

        # ====== receipt handling ======
        has_photo = bool(msg.get("photo"))
        has_doc = bool(msg.get("document"))
        if has_photo or has_doc:
            user_id = from_id
            if has_photo:
                file_id = msg["photo"][-1].get("file_id")
                kind = "photo"
            else:
                file_id = msg["document"].get("file_id")
                kind = "document"

            if file_id:
                async with SessionLocal() as session:
                    q = (
                        select(Order)
                        .where(
                            Order.user_id == user_id,
                            Order.status.in_([OrderStatus.AWAITING_PAYMENT.value, OrderStatus.REJECTED.value])
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
                    order.receipt_kind = kind
                    order.receipt_message_id = msg.get("message_id")
                    order.status = OrderStatus.RECEIPT_UPLOADED.value
                    order.updated_at = utcnow()
                    await session.commit()

                kb = {"inline_keyboard": [[{"text": "✅ Оплатил", "callback_data": f"paydone:{order.id}"}]]}
                await tg_call("sendMessage", {
                    "chat_id": user_id,
                    "text": f"📎 Чек по заказу `{order.id}` получен.\nНажми *✅ Оплатил*, чтобы отправить на проверку админу.",
                    "parse_mode": "Markdown",
                    "reply_markup": kb,
                })
                return {"ok": True}

        return {"ok": True}

    # =========================
    # CALLBACKS
    # =========================
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

    async with SessionLocal() as session:
        if data == "profile:open":
            u = await session.get(User, from_id)
            if not u:
                now = utcnow()
                u = User(user_id=from_id, full_name=None, phone=None, birth_date=None, is_registered=False,
                         created_at=now, updated_at=now, reg_bonus_deadline_at=None, reg_bonus_granted=False)
                session.add(u)
                await session.commit()

            wL = await get_or_create_wallet(session, from_id, WalletKind.LOYALTY.value)
            wR = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)

            reg = "✅ Да" if u.is_registered else "❌ Нет"
            txt = (
                "👤 *Профиль*\n\n"
                f"ID: `{u.user_id}`\n"
                f"Регистрация: {reg}\n"
                f"Имя: {u.full_name or '—'}\n"
                f"Телефон: {u.phone or '—'}\n"
                f"Дата рождения: {u.birth_date or '—'}\n\n"
                f"🎁 Баллы лояльности: *{wL.balance}*\n"
                f"🤝 Реферальные баллы: *{wR.balance}*\n"
            )
            kb = {
                "inline_keyboard": [
                    [{"text": "🛍 Каталог", "web_app": {"url": MINIAPP_URL}}],
                    [{"text": "✅ Регистрация", "callback_data": "reg:start"}] if not u.is_registered else [],
                    [{"text": "🎁 Лояльность", "callback_data": "loyalty:open"}],
                    [{"text": "🤝 Рефералка", "callback_data": "ref:open"}],
                ]
            }
            kb["inline_keyboard"] = [row for row in kb["inline_keyboard"] if row]
            await tg_call("sendMessage", {"chat_id": from_id, "text": txt, "parse_mode": "Markdown", "reply_markup": kb})
            return {"ok": True}

        if data == "menu:open":
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Меню:", "reply_markup": main_menu_keyboard()})
            return {"ok": True}

        if data == "reg:start":
            REG_STATE[from_id] = RegStep.NAME.value
            REG_TEMP[from_id] = {}
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите имя и фамилию:"})
            return {"ok": True}

        if data == "loyalty:open":
            u = await session.get(User, from_id)
            if not u or not u.is_registered:
                await tg_call("sendMessage", {
                    "chat_id": from_id,
                    "text": "Чтобы открыть Лояльность — нужно завершить регистрацию (имя, телефон, дата рождения).",
                    "reply_markup": {"inline_keyboard": [[{"text": "✅ Начать регистрацию", "callback_data": "reg:start"}]]}
                })
                return {"ok": True}

            c = await get_consent(session, from_id, ConsentKind.LOYALTY.value)
            if c.accepted is not True and not consent_can_show(c):
                mins = int((c.cooldown_until - utcnow()).total_seconds() // 60) if c.cooldown_until else 60
                await tg_call("sendMessage", {"chat_id": from_id, "text": f"Вы отклонили правила. Повторно можно через ~{mins} мин."})
                return {"ok": True}

            if c.accepted is not True:
                kb = {"inline_keyboard": [[
                    {"text": "✅ Принять", "callback_data": "loyalty:accept"},
                    {"text": "❌ Не принимать", "callback_data": "loyalty:decline"},
                ]]}
                await tg_call("sendMessage", {"chat_id": from_id, "text": rules_loyalty_text(), "parse_mode": "Markdown", "reply_markup": kb})
                return {"ok": True}

            w = await get_or_create_wallet(session, from_id, WalletKind.LOYALTY.value)
            txt = (
                "🎁 *Лояльность*\n\n"
                f"Баланс: *{w.balance}* баллов\n\n"
                "Действия:\n"
                "• Обменять баллы (2:1)\n"
                "• История (скоро)\n"
            )
            kb = {"inline_keyboard": [
                [{"text": "🔁 Обмен (Лояльность → Рефералка)", "callback_data": "ex:l2r"}],
                [{"text": "🔁 Обмен (Рефералка → Лояльность)", "callback_data": "ex:r2l"}],
                [{"text": "📜 История (скоро)", "callback_data": "noop"}],
                [{"text": "⬅️ Меню", "callback_data": "menu:open"}],
            ]}
            await tg_call("sendMessage", {"chat_id": from_id, "text": txt, "parse_mode": "Markdown", "reply_markup": kb})
            return {"ok": True}

        if data == "loyalty:accept":
            c = await get_consent(session, from_id, ConsentKind.LOYALTY.value)
            c.accepted = True
            c.cooldown_until = None
            c.updated_at = utcnow()
            await session.commit()
            await tg_call("sendMessage", {"chat_id": from_id, "text": "✅ Правила лояльности приняты."})
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Меню:", "reply_markup": main_menu_keyboard()})
            return {"ok": True}

        if data == "loyalty:decline":
            c = await get_consent(session, from_id, ConsentKind.LOYALTY.value)
            c.accepted = False
            c.cooldown_until = utcnow() + timedelta(hours=1)
            c.updated_at = utcnow()
            await session.commit()
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Ок. Кнопка «Лояльность» будет доступна снова через 1 час."})
            return {"ok": True}

        if data == "ref:open":
            u = await session.get(User, from_id)
            if not u or not u.is_registered:
                await tg_call("sendMessage", {
                    "chat_id": from_id,
                    "text": "Чтобы открыть Рефералку — нужно завершить регистрацию (имя, телефон, дата рождения).",
                    "reply_markup": {"inline_keyboard": [[{"text": "✅ Начать регистрацию", "callback_data": "reg:start"}]]}
                })
                return {"ok": True}

            c = await get_consent(session, from_id, ConsentKind.REFERRAL.value)
            if c.accepted is not True and not consent_can_show(c):
                mins = int((c.cooldown_until - utcnow()).total_seconds() // 60) if c.cooldown_until else 60
                await tg_call("sendMessage", {"chat_id": from_id, "text": f"Вы отклонили правила. Повторно можно через ~{mins} мин."})
                return {"ok": True}

            if c.accepted is not True:
                kb = {"inline_keyboard": [[
                    {"text": "✅ Принять", "callback_data": "ref:accept"},
                    {"text": "❌ Не принимать", "callback_data": "ref:decline"},
                ]]}
                await tg_call("sendMessage", {"chat_id": from_id, "text": rules_referral_text(), "parse_mode": "Markdown", "reply_markup": kb})
                return {"ok": True}

            code, link = await ensure_referral_assets(session, from_id)
            w = await get_or_create_wallet(session, from_id, WalletKind.REFERRAL.value)
            qcount = await count_qualified_referrals(session, from_id)
            pct = referral_percent_for_count(qcount)

            link_url = build_referral_tg_link(link.token)

            txt = (
                "🤝 *Рефералка*\n\n"
                f"Реферальные баллы: *{w.balance}* ₽\n"
                f"Ваш уровень: *{pct}%* (квалифицированных рефералов: {qcount})\n\n"
                f"Промокод: `{code.code}`\n"
                f"Ссылка: {link_url}\n\n"
                f"Вывод доступен от *{WITHDRAW_MIN_RUB} ₽*.\n"
            )
            kb = {"inline_keyboard": [
                [{"text": "🔁 Обмен (Лояльность → Рефералка)", "callback_data": "ex:l2r"}],
                [{"text": "🔁 Обмен (Рефералка → Лояльность)", "callback_data": "ex:r2l"}],
                [{"text": "💸 Запросить вывод (СБП)", "callback_data": "wd:start"}],
                [{"text": "📊 Мои рефералы (скоро)", "callback_data": "noop"}],
                [{"text": "⬅️ Меню", "callback_data": "menu:open"}],
            ]}
            await tg_call("sendMessage", {"chat_id": from_id, "text": txt, "parse_mode": "Markdown", "reply_markup": kb})
            return {"ok": True}

        if data == "ref:accept":
            c = await get_consent(session, from_id, ConsentKind.REFERRAL.value)
            c.accepted = True
            c.cooldown_until = None
            c.updated_at = utcnow()
            await session.commit()
            await tg_call("sendMessage", {"chat_id": from_id, "text": "✅ Правила реферальной программы приняты."})
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Меню:", "reply_markup": main_menu_keyboard()})
            return {"ok": True}

        if data == "ref:decline":
            c = await get_consent(session, from_id, ConsentKind.REFERRAL.value)
            c.accepted = False
            c.cooldown_until = utcnow() + timedelta(hours=1)
            c.updated_at = utcnow()
            await session.commit()
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Ок. Кнопка «Рефералка» будет доступна снова через 1 час."})
            return {"ok": True}

        if data == "ex:l2r":
            REG_STATE[from_id] = "EX_L2R"
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите количество баллов Лояльности для обмена (спишем 2:1). Например: 100"})
            return {"ok": True}

        if data == "ex:r2l":
            REG_STATE[from_id] = "EX_R2L"
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Введите количество реферальных баллов для обмена (спишем 2:1). Например: 100"})
            return {"ok": True}

        if data == "wd:start":
            REG_STATE[from_id] = "WD_AMOUNT"
            await tg_call("sendMessage", {"chat_id": from_id, "text": f"Введите сумму вывода (минимум {WITHDRAW_MIN_RUB})."})
            return {"ok": True}

        if data == "noop":
            await tg_call("sendMessage", {"chat_id": from_id, "text": "Скоро будет ✅"})
            return {"ok": True}

        # ----- existing admin/payment flow -----
        if data.startswith("choosepay:"):
            if not is_admin(from_id) or not GROUP_CHAT_ID:
                return {"ok": True}

            order_id = data.split("choosepay:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"⚠️ Заказ {order_id} не найден."})
                return {"ok": True}

            if order.status != OrderStatus.NEW.value:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"ℹ️ Заказ {order_id} уже не NEW (status={order.status})."})
                return {"ok": True}

            res = await session.execute(
                select(PaymentMethod)
                .where(PaymentMethod.is_active == True)  # noqa: E712
                .order_by(PaymentMethod.id.asc())
            )
            methods = res.scalars().all()

            if not methods:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": "⚠️ Нет активных реквизитов."})
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
                "chat_id": GROUP_CHAT_ID,
                "text": f"Выберите реквизит для заказа `{order_id}`:",
                "parse_mode": "Markdown",
                "reply_markup": {"inline_keyboard": buttons},
            })
            return {"ok": True}

        if data.startswith("setpay:"):
            if not is_admin(from_id) or not GROUP_CHAT_ID:
                return {"ok": True}

            _, order_id, mid = data.split(":", 2)
            method_id = int(mid)

            order = await session.get(Order, order_id)
            if not order or order.status != OrderStatus.NEW.value:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"⚠️ Нельзя назначить реквизит: заказ {order_id} не NEW."})
                return {"ok": True}

            method = await session.get(PaymentMethod, method_id)
            if not method or not method.is_active:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": "⚠️ Реквизит не найден/неактивен."})
                return {"ok": True}

            now = utcnow()
            order.payment_method_id = method.id
            order.accepted_by = from_id
            order.accepted_at = now
            order.status = OrderStatus.AWAITING_PAYMENT.value
            order.updated_at = now
            await session.commit()

            await tg_call("sendMessage", {
                "chat_id": order.user_id,
                "text": format_payment_to_user(order_id, method.title, method.text),
                "parse_mode": "Markdown",
            })

            await tg_call("sendMessage", {
                "chat_id": GROUP_CHAT_ID,
                "text": f"✅ Заказ `{order_id}`: выбран реквизит *{method.title}*. Реквизиты отправлены пользователю.",
                "parse_mode": "Markdown",
            })
            return {"ok": True}

        if data.startswith("paydone:"):
            if not GROUP_CHAT_ID:
                return {"ok": True}

            order_id = data.split("paydone:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Заказ не найден."})
                return {"ok": True}

            if from_id != order.user_id:
                return {"ok": True}

            if not order.receipt_file_id or order.status != OrderStatus.RECEIPT_UPLOADED.value:
                await tg_call("sendMessage", {"chat_id": from_id, "text": "Сначала отправь чек, потом нажми ✅ Оплатил."})
                return {"ok": True}

            order.status = OrderStatus.UNDER_REVIEW.value
            order.updated_at = utcnow()
            await session.commit()

            kb = {"inline_keyboard": [[
                {"text": "✅ Подтвердить", "callback_data": f"paid:{order.id}"},
                {"text": "❌ Отклонить", "callback_data": f"reject:{order.id}"},
            ]]}

            await tg_call("sendMessage", {
                "chat_id": GROUP_CHAT_ID,
                "text": f"🧾 *Проверка оплаты*\nЗаказ `{order.id}`\nПользователь `{order.user_id}`\nСумма: *{order.total} ₽*",
                "parse_mode": "Markdown",
                "reply_markup": kb,
            })

            try:
                if order.receipt_kind == "photo":
                    await tg_call("sendPhoto", {"chat_id": GROUP_CHAT_ID, "photo": order.receipt_file_id, "caption": f"Чек заказа {order.id}"})
                else:
                    await tg_call("sendDocument", {"chat_id": GROUP_CHAT_ID, "document": order.receipt_file_id, "caption": f"Чек заказа {order.id}"})
            except Exception as e:
                await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"⚠️ Не смог отправить чек: {repr(e)}"})

            await tg_call("sendMessage", {"chat_id": order.user_id, "text": f"✅ Заказ `{order.id}` отправлен на проверку.", "parse_mode": "Markdown"})
            return {"ok": True}

        if data.startswith("paid:"):
            if not is_admin(from_id) or not GROUP_CHAT_ID:
                return {"ok": True}

            order_id = data.split("paid:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}

            order.status = OrderStatus.PAID.value
            order.updated_at = utcnow()
            await session.commit()

            # 1) Loyalty cashback
            q = select(Order).where(Order.user_id == order.user_id, Order.status == OrderStatus.PAID.value)
            res = await session.execute(q)
            paid_orders = res.scalars().all()
            total_spend = sum(o.total for o in paid_orders)

            pct = loyalty_percent_for_spend(total_spend)
            cashback = max(0, int(order.total * pct / 100))
            if cashback > 0:
                await wallet_credit(session, order.user_id, WalletKind.LOYALTY.value, cashback, {"reason": "cashback", "order_id": order.id, "pct": pct})
                await session.commit()

            # 2) Referral payout
            qrel = select(ReferralRelation).where(ReferralRelation.referral_user_id == order.user_id)
            rres = await session.execute(qrel)
            rel = rres.scalar_one_or_none()
            if rel and not rel.is_detached:
                rel.referral_paid_orders += 1
                rel.referral_paid_total += int(order.total)
                rel.updated_at = utcnow()

                qualified = rel.referral_paid_total >= REFERRAL_TRIGGER_TOTAL

                # платим только если реферал уже квалифицирован и не исчерпан лимит выплат
                if qualified and rel.payouts_done_orders < REFERRAL_PAYOUT_ORDERS_LIMIT:
                    qcount = await count_qualified_referrals(session, rel.referrer_user_id)
                    pct_ref = min(referral_percent_for_count(qcount), REFERRAL_MAX_PERCENT)
                    payout = max(0, int(order.total * pct_ref / 100))
                    if payout > 0:
                        await wallet_credit(
                            session,
                            rel.referrer_user_id,
                            WalletKind.REFERRAL.value,
                            payout,
                            {"reason": "ref_payout", "order_id": order.id, "referral_user_id": order.user_id, "pct": pct_ref},
                        )
                        rel.payouts_done_orders += 1

                # отвязка после 3-го оплаченного заказа (как в ТЗ)
                if rel.referral_paid_orders >= REFERRAL_PAYOUT_ORDERS_LIMIT:
                    rel.is_detached = True

                await session.commit()

            # 3) Registration bonus / gate messaging
            u = await session.get(User, order.user_id)
            if not u:
                now = utcnow()
                u = User(
                    user_id=order.user_id,
                    full_name=None,
                    phone=None,
                    birth_date=None,
                    is_registered=False,
                    created_at=now,
                    updated_at=now,
                    reg_bonus_deadline_at=None,
                    reg_bonus_granted=False,
                )
                session.add(u)
                await session.flush()

            # Первый оплаченный заказ → запускаем окно бонуса 24ч (1 раз)
            if (not u.is_registered) and (u.reg_bonus_deadline_at is None):
                u.reg_bonus_deadline_at = utcnow() + timedelta(hours=REG_BONUS_WINDOW_HOURS)
                u.updated_at = utcnow()
                await session.commit()

                await tg_call("sendMessage", {
                    "chat_id": order.user_id,
                    "text": (
                        "🎁 Бонус за регистрацию!\n\n"
                        f"Заверши регистрацию (имя, телефон, дата рождения) и получи *{REG_BONUS_POINTS}* баллов Лояльности.\n"
                        f"У тебя есть *{REG_BONUS_WINDOW_HOURS} часа*.\n\n"
                        "Нажми кнопку ниже:"
                    ),
                    "parse_mode": "Markdown",
                    "reply_markup": {"inline_keyboard": [[{"text": "✅ Завершить регистрацию", "callback_data": "reg:start"}]]},
                })

            # Если окно 24ч уже прошло и он всё ещё не зарегистрирован:
            # на втором и последующих заказах предлагаем регистрацию БЕЗ бонуса, чтобы открыть функции
            if (not u.is_registered) and (u.reg_bonus_deadline_at is not None) and (utcnow() > u.reg_bonus_deadline_at):
                # определяем, что это минимум 2-й оплаченный заказ
                if len(paid_orders) >= 2:
                    await tg_call("sendMessage", {
                        "chat_id": order.user_id,
                        "text": (
                            "ℹ️ Чтобы открыть Лояльность / Рефералку / Профиль — нужно завершить регистрацию.\n"
                            "Бонус 2000 уже недоступен, но доступ к функциям откроется сразу после регистрации."
                        ),
                        "reply_markup": {"inline_keyboard": [[{"text": "✅ Завершить регистрацию", "callback_data": "reg:start"}]]},
                    })

            await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"✅ Оплата подтверждена по заказу `{order_id}`.", "parse_mode": "Markdown"})
            await tg_call("sendMessage", {"chat_id": order.user_id, "text": f"🎉 Оплата по заказу `{order_id}` подтверждена! ✅", "parse_mode": "Markdown"})
            return {"ok": True}

        if data.startswith("reject:"):
            if not is_admin(from_id) or not GROUP_CHAT_ID:
                return {"ok": True}

            order_id = data.split("reject:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}

            order.status = OrderStatus.REJECTED.value
            order.updated_at = utcnow()
            await session.commit()

            await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"❌ Чек отклонён по заказу `{order_id}`.", "parse_mode": "Markdown"})
            await tg_call("sendMessage", {"chat_id": order.user_id, "text": f"❌ Чек по заказу `{order_id}` отклонён. Пришли новый чек.", "parse_mode": "Markdown"})
            return {"ok": True}

        if data.startswith("cancel:"):
            if not is_admin(from_id) or not GROUP_CHAT_ID:
                return {"ok": True}

            order_id = data.split("cancel:", 1)[1].strip()
            order = await session.get(Order, order_id)
            if not order:
                return {"ok": True}

            order.status = OrderStatus.CANCELLED.value
            order.updated_at = utcnow()
            await session.commit()

            await tg_call("sendMessage", {"chat_id": GROUP_CHAT_ID, "text": f"❌ Заказ `{order_id}` отменён.", "parse_mode": "Markdown"})
            await tg_call("sendMessage", {"chat_id": order.user_id, "text": f"❌ Заказ `{order_id}` отменён админом.", "parse_mode": "Markdown"})
            return {"ok": True}

    return {"ok": True}