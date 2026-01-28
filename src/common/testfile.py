import pandas as pd
import os

file_path = "data/fact_vacancies_raw.csv"

# 1. Проверяем размер файла
if os.path.exists(file_path):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"💾 Вес файла на диске: {size_mb:.2f} MB")
else:
    print("❌ Файл не найден!")

# 2. Считаем строки через Pandas
try:
    df = pd.read_csv(file_path)
    print(f"📊 Количество строк (row count): {len(df)}")
    print(f"🆔 Уникальных vacancy_id: {df['vacancy_id'].nunique()}")
except Exception as e:
    print(f"Ошибка чтения: {e}")
