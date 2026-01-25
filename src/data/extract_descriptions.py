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
    primary = os.getenv("SRC_PG_DSN")
    if not primary: sys.exit("❌ Error: SRC_PG_DSN missing")
    print(f"🔌 Using POOLER connection (6543)...")
    return primary


def _connect(dsn: str):
    if "sslmode=" not in dsn: dsn += "&sslmode=require" if "?" in dsn else "?sslmode=require"
    try:
        conn = psycopg2.connect(
            dsn,
            cursor_factory=RealDictCursor,
            connect_timeout=20,  # Увеличили
            options="-c statement_timeout=30000"  # Увеличили до 30 сек
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        raise OperationalError(f"Connect failed: {e}")


def fetch_table(dsn: str, cfg: FetchConfig, limit_total: int) -> pd.DataFrame:
    all_data = []
    total_fetched = 0
    last_id: Optional[int] = None

    # СТАРТУЕМ С 1. ЭТО САМЫЙ БЕЗОПАСНЫЙ ВАРИАНТ.
    current_batch_size = 1

    # Обрезаем текст, чтобы пролезло
    cols_sql = "vacancy_id, LEFT(vacancy_description, 2000) as vacancy_description"

    print(f"📦 Fetching [{cfg.table_name}] (Target: {limit_total}, Batch: 1, Truncated)...")

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

                # Пишем каждый шаг, так как идем медленно
                print(f"   -> {total_fetched}/{limit_total}")

                # Даем сети отдышаться
                time.sleep(0.5)

            except Exception as e:
                attempt += 1
                print(f"   ⚠️ Error ({attempt}/{cfg.max_retries}): {str(e).splitlines()[0]}")
                time.sleep(cfg.base_sleep_s * attempt)
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass

    return pd.DataFrame(all_data)


def main():
    print("🚀 STEP 2: Extract Descriptions (Super Safe)...")
    root = get_project_root()
    load_dotenv(root / ".env")

    out_dir = root / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "descriptions.parquet"

    dsn = _get_dsn()
    limit = int(os.getenv("SRC_TARGET_ROWS", "2000"))

    desc_cfg = FetchConfig(
        table_name=os.getenv("SRC_DIM_TABLE", "public.dim_vacancy_descriptions_versioned"),
        columns=[],
        batch_size=1,  # Игнорируется, так как переопределено внутри
        min_batch_size=1, max_retries=20, base_sleep_s=2.0, backoff=1.5
    )

    df = fetch_table(dsn, desc_cfg, limit)

    if not df.empty:
        df.to_parquet(out_file, index=False)
        print(f"✅ Descriptions saved to: {out_file}")
        print(f"📊 Count: {len(df)}")
    else:
        print("❌ No descriptions downloaded.")


if __name__ == "__main__":
    main()
