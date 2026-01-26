import pandas as pd
import numpy as np
import pathlib
from html.parser import HTMLParser

# Настройка путей
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "dataset"
DATA_DIR.mkdir(exist_ok=True)


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


def strip_tags(html_txt):
    if not isinstance(html_txt, str): return ""
    s = MLStripper()
    s.feed(html_txt)
    return " ".join(s.get_data().split())


def parse_pg_array(array_str):
    """Парсит строку '{1,2,3}' в список [1, 2, 3]"""
    if pd.isna(array_str) or str(array_str) == '{}': return []
    content = str(array_str).strip('{}')
    if not content: return []
    return [int(x) for x in content.split(',') if x.strip().isdigit()]


def prepare_dataset(input_csv: str, output_parquet: str):
    print(f"📥 Загрузка данных из {input_csv}...")

    # 1. Загрузка основных данных
    df = pd.read_csv(input_csv)

    # 2. Загрузка справочника навыков (переводчик из ID в текст)
    skills_map_path = ROOT_DIR / "skills.csv"
    if skills_map_path.exists():
        print("🔗 Загрузка справочника навыков skills.csv...")
        df_skills = pd.read_csv(skills_map_path)
        skill_map = dict(zip(df_skills['skill_id'], df_skills['name']))
    else:
        print("⚠️ ВНИМАНИЕ: skills.csv не найден! Навыки будут пустыми.")
        skill_map = {}

    # 3. Обработка дат и Velocity
    df['upd_date'] = pd.to_datetime(df['last_update_date'])
    df['pub_date'] = pd.to_datetime(df['publication_date'])

    # Считаем время жизни (минимум 0.5 дня, чтобы не делить на 0)
    df['days_live'] = (df['upd_date'] - df['pub_date']).dt.total_seconds() / (24 * 3600)
    df['days_live'] = df['days_live'].apply(lambda x: max(x, 0.5))

    # Velocity: сколько откликов в день
    df['total_responses'] = df['total_responses'].fillna(0)
    df['velocity'] = df['total_responses'] / df['days_live']

    # 4. Определение Top Performers (Успешные)
    # Если данных мало (например, выгрузка за 2 дня), мы снижаем планку
    # Вакансия считается успешной, если у неё > 0.1 отклика в день
    df['is_top_performer'] = df['velocity'] > 0.1

    # Если все равно 0 успешных, берем просто топ 10% самых активных
    if df['is_top_performer'].sum() == 0:
        threshold = df['velocity'].quantile(0.9)
        df['is_top_performer'] = df['velocity'] >= threshold

    print(f"🏆 Найдено эталонных вакансий: {df['is_top_performer'].sum()} из {len(df)}")

    # 5. Обработка текстов и навыков
    print("🧹 Очистка текста и маппинг навыков...")
    df['text_clean'] = df['vacancy_description'].apply(strip_tags)

    # Превращаем ID {1,2} в текст "Навык1, Навык2"
    def map_skills(val):
        ids = parse_pg_array(val)
        return ", ".join([skill_map.get(i, "") for i in ids if i in skill_map])

    # В CSV колонка называется skill_ids (из вашего SQL)
    if 'skill_ids' in df.columns:
        df['skills_str'] = df['skill_ids'].apply(map_skills)
    else:
        df['skills_str'] = ""

    # 6. Сохранение
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

    df[final_cols].to_parquet(output_parquet, index=False)
    print(f"✅ Готовый датасет сохранен: {output_parquet}")


if __name__ == "__main__":
    prepare_dataset('vacancies_export.csv', 'dataset/vacancies_processed.parquet')
