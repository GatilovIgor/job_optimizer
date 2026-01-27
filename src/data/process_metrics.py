import pandas as pd
import numpy as np
import pathlib
from datetime import datetime
import sys

# Добавляем путь к корню, чтобы видеть src.common
# (если запускаете через python -m, это может быть не обязательно, но надежно)
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Импортируем вашу функцию очистки
try:
    from src.common.text import normalize_text
except ImportError:
    # Если запуск не из корня, запасной вариант
    import re


    def normalize_text(t):
        t = re.sub(r"<[^>]+>", " ", str(t))
        return re.sub(r"\s+", " ", t).strip()

# --- CONFIG ---
MIN_DAYS_ACTIVE = 1
TOP_QUANTILE = 0.80


def get_project_root() -> pathlib.Path:
    current_path = pathlib.Path(__file__).resolve().parent
    for _ in range(5):
        if (current_path / ".env").exists(): return current_path
        current_path = current_path.parent
    return pathlib.Path.cwd()


def main():
    print("🧠 Processing Data for ML (using common.text)...")
    root = get_project_root()

    input_file = root / "data" / "vacancies_full.parquet"
    output_file = root / "data" / "vacancies_processed.parquet"

    if not input_file.exists():
        print(f"❌ Input not found: {input_file}")
        return

    df = pd.read_parquet(input_file)
    print(f"   Loaded: {len(df)} rows")

    # 1. Формируем текст
    print("   Cleaning text using src.common.text.normalize_text...")
    df['full_raw_text'] = df['vacancy_title'] + " " + df['vacancy_description'].fillna("")

    # ИСПОЛЬЗУЕМ ВАШУ ФУНКЦИЮ
    df['text_clean'] = df['full_raw_text'].apply(lambda x: normalize_text(str(x)))

    # 2. Метрики (Velocity)
    now = datetime.now()
    df['start_date'] = df['publication_date'].fillna(df['creation_date'])
    df['days_active'] = (now - df['start_date']).dt.days.clip(lower=1)
    df['velocity'] = df['total_responses'] / df['days_active']

    # 3. Чемпионы
    threshold = df['velocity'].quantile(TOP_QUANTILE)
    df['is_top_performer'] = df['velocity'] >= threshold

    print(f"   🏆 Top Threshold: > {threshold:.2f} responses/day")
    print(f"   Found {df['is_top_performer'].sum()} top performers.")

    # 4. Сохранение
    cols = ['vacancy_id', 'vacancy_title', 'text_clean', 'velocity', 'is_top_performer']
    df[cols].to_parquet(output_file, index=False)

    print(f"✅ Saved processed data: {output_file}")


if __name__ == "__main__":
    main()
