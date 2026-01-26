import pandas as pd
import numpy as np
import re
import pathlib
from datetime import datetime

# --- CONFIG ---
MIN_DAYS_ACTIVE = 1  # Для демо можно и 1 день
TOP_QUANTILE = 0.80  # Топ-20% лучших


def get_project_root() -> pathlib.Path:
    current_path = pathlib.Path(__file__).resolve().parent
    for _ in range(5):
        if (current_path / ".env").exists(): return current_path
        current_path = current_path.parent
    return pathlib.Path.cwd()


def clean_text(text):
    if not isinstance(text, str): return ""
    # Убираем HTML заглушки, оставляем только суть
    # (В вашем случае там заглушки, но на будущее полезно)
    text = re.sub(r'<.*?>', ' ', text)
    return text.strip()


def main():
    print("🧠 Processing Data for ML...")
    root = get_project_root()

    input_file = root / "dataset" / "vacancies_full.parquet"
    output_file = root / "dataset" / "vacancies_processed.parquet"

    if not input_file.exists():
        print(f"❌ Input not found: {input_file}")
        return

    df = pd.read_parquet(input_file)
    print(f"   Loaded: {len(df)} rows")

    # 1. Формируем текст для поиска
    # Используем Title как основной сигнал (так как описания - заглушки)
    # Если бы были реальные описания, мы бы чистили их тут.
    df['text_clean'] = df['vacancy_title'].apply(clean_text)

    # 2. Считаем Velocity (Метрика успеха)
    now = datetime.now()
    df['start_date'] = df['publication_date'].fillna(df['creation_date'])

    # Избегаем деления на ноль
    df['days_active'] = (now - df['start_date']).dt.days.clip(lower=1)

    # Velocity = Отклики / Дни
    df['velocity'] = df['total_responses'] / df['days_active']

    # 3. Определяем "Чемпионов" (Top Performers)
    threshold = df['velocity'].quantile(TOP_QUANTILE)
    df['is_top_performer'] = df['velocity'] >= threshold

    print(f"   🏆 Top Threshold: > {threshold:.2f} responses/day")
    print(f"   Found {df['is_top_performer'].sum()} top performers.")

    # 4. Сохраняем
    cols = ['vacancy_id', 'vacancy_title', 'text_clean', 'velocity', 'is_top_performer']
    df[cols].to_parquet(output_file, index=False)

    print(f"✅ Saved processed data: {output_file}")


if __name__ == "__main__":
    main()
