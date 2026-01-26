import pandas as pd
import numpy as np
import json
from html.parser import HTMLParser
from datetime import datetime


# --- Утилита для очистки HTML ---
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, d):
        self.text.append(d)

    def get_data(self):
        return "".join(self.text)


def strip_tags(html):
    if not isinstance(html, str): return ""
    s = MLStripper()
    s.feed(html)
    return " ".join(s.get_data().split())


def prepare_dataset(input_csv: str, output_parquet: str):
    print(f"📥 Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Парсинг дат
    # Используем last_update_date как "текущий момент" для снэпшота
    now_date = pd.to_datetime(df['last_update_date']).max()
    df['pub_date'] = pd.to_datetime(df['publication_date'])

    # Считаем время жизни вакансии в днях (минимум 1 день, чтобы не делить на 0)
    df['days_live'] = (now_date - df['pub_date']).dt.total_seconds() / (24 * 3600)
    df['days_live'] = df['days_live'].apply(lambda x: max(x, 1))

    # 2. Расчет Velocity (Откликов в день)
    # Заменяем total_responses NaN на 0
    df['total_responses'] = df['total_responses'].fillna(0)
    df['velocity'] = df['total_responses'] / df['days_live']

    # 3. Определение Top Performer (Успешная вакансия)
    # Логика: Вакансия успешна, если её скорость выше средней по этому профилю
    # Считаем Z-score для скорости внутри каждого Profile

    # Сначала сгруппируем и посчитаем статистики
    profile_stats = df.groupby('profile')['velocity'].agg(['mean', 'std']).reset_index()
    df = df.merge(profile_stats, on='profile', suffixes=('', '_stats'))

    # Если std = 0 (одна вакансия в профиле), z_score = 0
    df['velocity_z'] = (df['velocity'] - df['mean']) / df['std'].replace(0, 1)

    # Условие успеха: Топ 30% (Z-score > 0.5) ИЛИ просто много откликов (> 1 в день)
    # Фильтруем совсем новые вакансии (< 3 дней), чтобы не вносить шум
    df['is_top_performer'] = (
            ((df['velocity_z'] > 0.5) | (df['velocity'] > 1.0)) &
            (df['days_live'] >= 3)
    )

    print(f"🏆 Identified {df['is_top_performer'].sum()} top performers out of {len(df)} vacancies.")

    # 4. Обработка текста
    print("🧹 Cleaning text...")
    df['text_clean'] = df['vacancy_description'].apply(strip_tags)

    # Обработка навыков (JSON -> String)
    def clean_skills(x):
        try:
            return " ".join(json.loads(x))
        except:
            return ""

    df['skills_str'] = df['skills'].apply(clean_skills)

    # 5. Отбор колонок для RAG
    # Нам нужны:
    # - vacancy_title, skills_str, specialization, text_clean (для поиска)
    # - vacancy_description (RAW HTML для LLM, чтобы она училась форматированию)
    # - velocity (для сортировки)
    # - is_top_performer (для фильтра)

    final_cols = [
        'vacancy_title',
        'vacancy_description',
        'text_clean',
        'skills_str',
        'specialization',
        'profile',
        'velocity',
        'is_top_performer'
    ]

    # Сохраняем
    df[final_cols].to_parquet(output_parquet, index=False)
    print(f"✅ Saved processed dataset to {output_parquet}")


if __name__ == "__main__":
    # Запуск: python src/data/prepare.py
    prepare_dataset('vacancies_export.csv', 'dataset/vacancies_processed.parquet')
