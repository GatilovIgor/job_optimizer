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

# Сохраняем в отдельный файл, чтобы не затереть основные данные
OUTPUT_FILE = DATA_DIR / "fact_vacancies_test.csv"

# --- 2. Настройки таблицы ---
# Жестко прописываем имя тестовой таблицы
TARGET_TABLE = "fact_vacancies_cleaned_test"

# --- 3. Загрузка конфига (.env) ---
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path)


def load_test_data():
    print(f"--- 🧪 Старт загрузки ТЕСТОВЫХ данных ({TARGET_TABLE}) ---")

    # Берем подключение из того же .env
    db_dsn = os.getenv("DB_DSN")

    if not db_dsn:
        print("❌ ОШИБКА: DB_DSN не найден в .env")
        sys.exit(1)

    try:
        engine = create_engine(db_dsn)

        # Шаг 1: Считаем количество строк
        print("📊 Подсчет количества строк...")
        with engine.connect() as conn:
            # Используем text() для безопасного SQL
            count_query = text(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
            total_rows = conn.execute(count_query).scalar()

        print(f"Всего строк в тесте: {total_rows}")

        if total_rows == 0:
            print("⚠️ Тестовая таблица пуста.")
            return

        # Шаг 2: Скачиваем с прогресс-баром
        chunk_size = 5000

        with tqdm(total=total_rows, unit="row", desc="Скачивание теста") as pbar:
            chunks = pd.read_sql(f"SELECT * FROM {TARGET_TABLE}", engine, chunksize=chunk_size)

            for i, chunk in enumerate(chunks):
                mode = 'w' if i == 0 else 'a'
                header = (i == 0)

                chunk.to_csv(OUTPUT_FILE, mode=mode, index=False, header=header)
                pbar.update(len(chunk))

        print(f"\n✅ Тестовые данные сохранены: {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")


if __name__ == "__main__":
    load_test_data()
