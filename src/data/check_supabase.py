from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv
from psycopg import OperationalError


@dataclass(frozen=True)
class RetryCfg:
    batch_size: int
    min_batch_size: int
    max_retries: int
    base_sleep_s: float
    backoff: float


def _normalize_dsn(raw: str) -> str:
    dsn = raw.replace("postgresql+psycopg://", "postgresql://", 1)
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn


def _sanitize_dsn(dsn: str) -> str:
    # show scheme://user@host:port/db?...
    p = urlparse(dsn)
    user = p.username or ""
    host = p.hostname or ""
    port = p.port or ""
    db = (p.path or "").lstrip("/")
    return f"{p.scheme}://{user}@{host}:{port}/{db}?..."


def _get_src_dsn() -> str:
    primary = os.getenv("SRC_PG_DSN")
    if not primary:
        sys.exit("❌ SRC_PG_DSN is not set")

    direct = os.getenv("SRC_PG_DSN_DIRECT")
    if direct:
        return _normalize_dsn(direct)

    primary = _normalize_dsn(primary)
    return primary


def _connect(dsn: str) -> psycopg.Connection:
    # prepare_threshold=0 отключает server-side prepare (меньше сюрпризов с прокси/пулерами)
    # options: снимаем клиентские таймауты на уровне сессии (на стороне Supabase могут быть лимиты)
    return psycopg.connect(
        dsn,
        prepare_threshold=0,
        options="-c statement_timeout=0 -c idle_in_transaction_session_timeout=0",
    )


def _run_one(dsn: str, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
    with _connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def _fetch_batch(
    dsn: str,
    table: str,
    cols_sql: str,
    last_id: Optional[str],
    batch_size: int,
) -> list[tuple]:
    if last_id is None:
        sql = f"""
            select {cols_sql}
            from {table}
            order by vacancy_id
            limit %s
        """
        params: Tuple[Any, ...] = (batch_size,)
    else:
        sql = f"""
            select {cols_sql}
            from {table}
            where vacancy_id > %s
            order by vacancy_id
            limit %s
        """
        params = (last_id, batch_size)

    with _connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _sleep(cfg: RetryCfg, attempt: int) -> None:
    delay = cfg.base_sleep_s * (cfg.backoff ** max(0, attempt - 1))
    time.sleep(delay)


def keyset_fetch_total(
    dsn: str,
    table: str,
    cols_sql: str,
    target_rows: int,
    cfg: RetryCfg,
) -> int:
    total = 0
    last_id: Optional[str] = None

    batch_size = cfg.batch_size
    attempt = 0

    while total < target_rows:
        take = min(batch_size, target_rows - total)

        try:
            rows = _fetch_batch(dsn, table, cols_sql, last_id, take)
            attempt = 0  # reset on success

            if not rows:
                break

            total += len(rows)
            last_id = rows[-1][0]  # vacancy_id должен быть первой колонкой

            if total % 250 == 0 or total == target_rows:
                print(f"...fetched {total}")

        except OperationalError as e:
            attempt += 1
            print(f"⚠️  OperationalError (attempt {attempt}/{cfg.max_retries}): {e}")

            if batch_size > cfg.min_batch_size:
                batch_size = max(cfg.min_batch_size, batch_size // 2)
                print(f"   reducing batch_size -> {batch_size}")

            if attempt >= cfg.max_retries:
                raise

            _sleep(cfg, attempt)

    return total


def main() -> None:
    load_dotenv()

    dsn = _get_src_dsn()
    table = os.getenv("SRC_FACT_TABLE", "public.fact_vacancies_versioned")

    cfg = RetryCfg(
        batch_size=int(os.getenv("SRC_BATCH_SIZE", "25")),
        min_batch_size=int(os.getenv("SRC_MIN_BATCH_SIZE", "5")),
        max_retries=int(os.getenv("SRC_MAX_RETRIES", "8")),
        base_sleep_s=float(os.getenv("SRC_RETRY_SLEEP_S", "0.75")),
        backoff=float(os.getenv("SRC_RETRY_BACKOFF", "2.0")),
    )

    target = int(os.getenv("SRC_TARGET_ROWS", "2000"))

    print("DSN (sanitized):", _sanitize_dsn(dsn))
    print("Fact table:", table)

    print("\n=== SELECT 1 ===")
    print(_run_one(dsn, "select 1"))

    print("\n=== CHECK TABLE EXISTS ===")
    print(_run_one(dsn, "select to_regclass(%s)", (table,)))

    print("\n=== LIMIT 1 (required columns) ===")
    print(
        _run_one(
            dsn,
            f"""
            select
                vacancy_id,
                vacancy_title,
                total_responses,
                publication_date,
                creation_date,
                last_update_date
            from {table}
            limit 1
            """,
        )
    )

    print(f"\n=== KEYSET FETCH {target} (batch={cfg.batch_size}, min={cfg.min_batch_size}) ===")
    t0 = time.time()
    cols_sql = "vacancy_id, vacancy_title, total_responses"
    n = keyset_fetch_total(dsn, table, cols_sql, target, cfg)
    print(f"\n✅ fetched {n} rows in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
