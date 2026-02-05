import os
import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text as sql_text


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)

    # ⚠️ ДАННЫЕ УДАЛЯЮТСЯ
    drop_sql = """
    DROP TABLE IF EXISTS orders CASCADE;
    DROP TABLE IF EXISTS payment_methods CASCADE;
    """

    # создаём ровно те таблицы/колонки, которые ждёт текущий main.py
    create_sql = """
    CREATE TABLE IF NOT EXISTS payment_methods (
        id SERIAL PRIMARY KEY,
        title VARCHAR(64) NOT NULL,
        text TEXT NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS orders (
        id VARCHAR(32) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        items JSON NOT NULL DEFAULT '{}'::json,
        total INTEGER NOT NULL DEFAULT 0,
        status VARCHAR(32) NOT NULL DEFAULT 'NEW',

        payment_method_id INTEGER NULL,
        accepted_by BIGINT NULL,
        accepted_at TIMESTAMPTZ NULL,

        receipt_file_id TEXT NULL,
        receipt_kind VARCHAR(16) NULL,
        receipt_message_id BIGINT NULL,

        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX IF NOT EXISTS ix_orders_user_id ON orders(user_id);
    """

    async with engine.begin() as conn:
        await conn.execute(sql_text(drop_sql))
        await conn.execute(sql_text(create_sql))

    await engine.dispose()
    print("✅ Rebuild done: orders + payment_methods recreated")


if __name__ == "__main__":
    asyncio.run(main())
