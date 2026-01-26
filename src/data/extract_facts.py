from __future__ import annotations
import os
import sys
import time
import pandas as pd
import pathlib
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError


# --- CONFIGURATION (COPY FROM WORKING CODE) ---

@dataclass
class FetchConfig:
    table_name: str
    columns: List[str]
    batch_size: int
    min_batch_size: int
    max_retries: int
    base_sleep_s: float
    backoff: float


def get_project_root() -> pathlib.Path:
    current_path = pathlib.Path(__file__).resolve().parent
    for _ in range(5):
        if (current_path / ".env").exists(): return current_path
        current_path = current_path.parent
    return pathlib.Path.cwd()


def _get_dsn() -> str:
    direct = os.getenv("SRC_PG_DSN_DIRECT")
    if direct and "#" not in direct:
        print("🔌 Using DIRECT connection (5432)...")
        return direct
    primary = os.getenv("SRC_PG_DSN")
    if not primary: sys.exit("❌ Error: SRC_PG_DSN missing")
    print(f"🔌 Using POOLER connection (6543)...")
    return primary


def _connect(dsn: str):
    # ТОЧНАЯ КОПИЯ ВАШЕГО РАБОЧЕГО КОННЕКТОРА
    if "sslmode=" not in dsn:
        dsn += "&sslmode=require" if "?" in dsn else "?sslmode=require"
    try:
        conn = psycopg2.connect(
            dsn,
            cursor_factory=RealDictCursor,
            connect_timeout=10,
            options="-c statement_timeout=15000"
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        raise OperationalError(f"Connect failed: {e}")


def fetch_table(dsn: str, cfg: FetchConfig, limit_total: int) -> pd.DataFrame:
    # ТОЧНАЯ КОПИЯ ВАШЕЙ ФУНКЦИИ
    all_data = []
    total_fetched = 0
    last_id: Optional[int] = None
    current_batch_size = cfg.batch_size
    cols_sql = ", ".join(cfg.columns)

    print(f"📦 Fetching [{cfg.table_name}] (Target: {limit_total})...")

    while total_fetched < limit_total:
        take = min(current_batch_size, limit_total - total_fetched)

        if last_id is None:
            sql = f"SELECT {cols_sql} FROM {cfg.table_name} ORDER BY vacancy_id LIMIT %s"
            params = (take,)
        else:
            sql = f"SELECT {cols_sql} FROM {cfg.table_name} WHERE vacancy_id > %s ORDER BY vacancy_id LIMIT %s"
            params = (last_id, take)

        attempt = 0
        success = False

        while not success and attempt < cfg.max_retries:
            conn = None
            try:
                conn = _connect(dsn)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()

                if not rows:
                    print(f"   🏁 End of table.")
                    if conn: conn.close()
                    return pd.DataFrame(all_data)

                all_data.extend(rows)
                total_fetched += len(rows)
                last_id = rows[-1]['vacancy_id']
                success = True

                if current_batch_size <= 5 or total_fetched % 50 == 0:
                    print(f"   -> {total_fetched}/{limit_total} (batch={current_batch_size})")
                time.sleep(0.1)

            except Exception as e:
                attempt += 1
                print(f"   ⚠️ Error ({attempt}/{cfg.max_retries}): {str(e).splitlines()[0]}")
                if current_batch_size > cfg.min_batch_size:
                    current_batch_size = max(cfg.min_batch_size, current_batch_size // 2)
                    print(f"      📉 Batch reduced -> {current_batch_size}")
                time.sleep(cfg.base_sleep_s * attempt)
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass

    return pd.DataFrame(all_data)


def main():
    print("🚀 STEP 1: Extract Facts...")
    root = get_project_root()
    load_dotenv(root / ".env")

    out_dir = root / "dataset"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "facts.parquet"

    dsn = _get_dsn()
    limit = int(os.getenv("SRC_TARGET_ROWS", "2000"))

    # Конфиг из вашего кода (только для фактов)
    fact_cfg = FetchConfig(
        table_name=os.getenv("SRC_FACT_TABLE", "public.fact_vacancies_versioned"),
        columns=["vacancy_id", "vacancy_title", "total_responses", "publication_date", "creation_date"],
        batch_size=25, min_batch_size=5, max_retries=10, base_sleep_s=1.0, backoff=1.5
    )

    df = fetch_table(dsn, fact_cfg, limit)

    if not df.empty:
        df.to_parquet(out_file, index=False)
        print(f"✅ Facts saved to: {out_file}")
        print(f"📊 Count: {len(df)}")
    else:
        print("❌ No facts downloaded.")


if __name__ == "__main__":
    main()
