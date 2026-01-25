import os
import time
import psycopg
from dotenv import load_dotenv


def test_dsn(name, dsn_raw):
    print(f"\n📡 Тестируем {name}...")
    if not dsn_raw:
        print("   ❌ Не задан в .env")
        return False

    # Чистим DSN
    dsn = dsn_raw.replace("postgresql+psycopg://", "postgresql://", 1)
    if "sslmode=" not in dsn: dsn += "&sslmode=require"

    # Ставим жесткий таймаут 5 секунд, чтобы не ждать вечно
    try:
        start = time.time()
        # connect_timeout=5 в параметрах строки
        conn_dsn = f"{dsn}&connect_timeout=5"

        with psycopg.connect(conn_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                res = cur.fetchone()
                print(f"   ✅ УСПЕХ! Ответ базы: {res}")
                print(f"   ⏱ Пинг: {(time.time() - start) * 1000:.1f} ms")
                return True
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        return False


def main():
    load_dotenv()

    direct = os.getenv("SRC_PG_DSN_DIRECT")  # Порт 5432
    pooler = os.getenv("SRC_PG_DSN")  # Порт 6543 (обычно надежнее)

    print("=== ЗАПУСК ДИАГНОСТИКИ СЕТИ ===")

    works_direct = test_dsn("DIRECT (5432)", direct)
    works_pooler = test_dsn("POOLER (6543)", pooler)

    print("\n=== ИТОГ ===")
    if works_direct:
        print("👉 Порт 5432 работает. Странно, что скрипт завис. Возможно, проблема в 'statement_timeout=0'.")
    elif works_pooler:
        print("👉 Порт 5432 БЛОКИРОВАН. Порт 6543 РАБОТАЕТ.")
        print("💡 РЕШЕНИЕ: Закомментируйте строку SRC_PG_DSN_DIRECT в файле .env")
    else:
        print("💀 Оба порта недоступны. Проверьте интернет или VPN.")


if __name__ == "__main__":
    main()
