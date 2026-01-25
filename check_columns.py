import os
import psycopg2
from dotenv import load_dotenv


def main():
    load_dotenv()
    dsn = os.getenv("SRC_PG_DSN")
    # Добавляем хаки для пулера (без них не законнектится)
    if "sslmode=" not in dsn: dsn += "&sslmode=require"

    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = True

    table = os.getenv("SRC_DIM_TABLE", "public.dim_vacancy_descriptions_versioned")
    # Убираем схему public. если есть, для запроса к information_schema
    table_name = table.split('.')[-1]

    sql = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}';
    """

    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [r[0] for r in cur.fetchall()]

    print(f"📋 Columns in {table}:")
    print(cols)


if __name__ == "__main__":
    main()
