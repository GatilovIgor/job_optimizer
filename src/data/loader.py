import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# --- 1. Настройка путей ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "fact_vacancies_raw.csv"

# --- 2. Загрузка конфига ---
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)


def load_data():
    print("--- 🚀 Старт загрузки (Режим стриминга) ---")

    db_dsn = os.getenv("DB_DSN")
    table_name = os.getenv("DB_TABLE_NAME", "fact_vacancies_cleaned")

    if not db_dsn:
        print("❌ ОШИБКА: DB_DSN не найден в .env")
        sys.exit(1)

    try:
        # Создаем engine
        engine = create_engine(db_dsn)

        # 1. Считаем количество (быстрое соединение)
        print("1️⃣  Считаем количество строк...")
        with engine.connect() as conn:
            count_query = text(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = conn.execute(count_query).scalar()

        print(f"📊 Всего строк: {total_rows}")

        # 2. Настраиваем потоковую выгрузку
        # stream_results=True ОБЯЗАТЕЛЕН, чтобы не грузить память
        conn = engine.connect().execution_options(stream_results=True)

        chunk_size = 2000

        print(f"2️⃣  Начинаю скачивание (по {chunk_size} строк)...")
        print("    Если сейчас не пойдет — значит интернет совсем плохой.")

        # Передаем conn (соединение), а не engine
        chunks = pd.read_sql(
            text(f"SELECT * FROM {table_name}"),
            conn,
            chunksize=chunk_size
        )

        with tqdm(total=total_rows, unit="row", desc="Скачивание") as pbar:
            for i, chunk in enumerate(chunks):
                mode = 'w' if i == 0 else 'a'
                header = (i == 0)

                chunk.to_csv(OUTPUT_FILE, mode=mode, index=False, header=header)
                pbar.update(len(chunk))

        # Не забываем закрыть соединение
        conn.close()
        print(f"\n✅ Готово! Файл сохранен: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\n🛑 Прервано пользователем.")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")


if __name__ == "__main__":
    load_data()
