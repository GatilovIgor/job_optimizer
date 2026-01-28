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

# Имя итогового файла
OUTPUT_FILE = DATA_DIR / "fact_vacancies_test.csv"

# --- 2. Загрузка конфига ---
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)


def load_test_data():
    print("--- 🧪 Старт загрузки ТЕСТОВЫХ данных ---")

    db_dsn = os.getenv("DB_DSN")
    # Жестко задаем имя тестовой таблицы
    table_name = "fact_vacancies_cleaned_test"

    if not db_dsn:
        print("❌ ОШИБКА: DB_DSN не найден в .env")
        sys.exit(1)

    try:
        # Создаем engine
        engine = create_engine(db_dsn)

        # 1. Считаем количество
        print(f"1️⃣  Считаем количество строк в таблице {table_name}...")
        try:
            with engine.connect() as conn:
                count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = conn.execute(count_query).scalar()
            print(f"📊 Всего строк: {total_rows}")
        except Exception as e:
            print(f"❌ Ошибка при чтении таблицы {table_name}. Возможно, её нет в БД?\nОшибка: {e}")
            return

        if total_rows == 0:
            print("⚠️ Таблица пуста. Файл не будет создан.")
            return

        # 2. Настраиваем потоковую выгрузку
        conn = engine.connect().execution_options(stream_results=True)
        chunk_size = 2000

        print(f"2️⃣  Начинаю скачивание в {OUTPUT_FILE.name}...")

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

        conn.close()
        print(f"\n✅ Успешно! Тестовый файл сохранен: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\n🛑 Прервано пользователем.")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")


if __name__ == "__main__":
    load_test_data()
