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
    if pd.isna(array_str) or str(array_str) == '{}': return []
    content = str(array_str).strip('{}')
    return [int(x) for x in content.split(',') if x.strip().isdigit()]


def prepare_dataset(input_csv: str, output_parquet: str):
    print(f"📥 Загрузка данных из {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Загрузка навыков
    skills_map_path = ROOT_DIR / "skills.csv"
    skill_map = {}
    if skills_map_path.exists():
        df_skills = pd.read_csv(skills_map_path)
        skill_map = dict(zip(df_skills['skill_id'], df_skills['name']))

    # 2. Обработка дат и расчет Velocity
    df['upd_date'] = pd.to_datetime(df['last_update_date'])
    df['pub_date'] = pd.to_datetime(df['publication_date'])
    # Считаем время жизни (минимум 0.1 дня для совсем новых)
    df['days_live'] = (df['upd_date'] - df['pub_date']).dt.total_seconds() / (24 * 3600)
    df['days_live'] = df['days_live'].apply(lambda x: max(x, 0.1))

    df['total_responses'] = df['total_responses'].fillna(0)
    df['velocity'] = df['total_responses'] / df['days_live']

    # 3. ГИБКИЙ ФИЛЬТР УСПЕХА
    print("🏆 Отбор эталонных вакансий...")

    # Считаем порог 80-го перцентиля (топ-20%) по всей базе
    velocity_threshold = df['velocity'].quantile(0.8)

    # Если все вакансии имеют 0 откликов, порог будет 0. В этом случае берем просто топ по длине текста.
    if velocity_threshold == 0:
        print("⚠️ Мало данных об откликах, отбираю по качеству описания...")
        df['is_top_performer'] = df['vacancy_description'].str.len() > df['vacancy_description'].str.len().median()
    else:
        # Условие: выше порога И текст больше 300 символов
        df['is_top_performer'] = (df['velocity'] >= velocity_threshold) & (df['vacancy_description'].str.len() > 300)

    # Если после фильтров всё еще 0, берем просто топ-100 по скорости
    if df['is_top_performer'].sum() == 0:
        df.loc[df.nlargest(100, 'velocity').index, 'is_top_performer'] = True

    # 4. Обработка текстов и навыков
    print("🧹 Очистка данных и маппинг навыков...")
    df['text_clean'] = df['vacancy_description'].apply(strip_tags)

    def map_skills(val):
        ids = parse_pg_array(val)
        return ", ".join([skill_map.get(i, "") for i in ids if i in skill_map])

    if 'skill_ids' in df.columns:
        df['skills_str'] = df['skill_ids'].apply(map_skills)
    else:
        df['skills_str'] = ""

    # 5. Сохранение
    final_cols = ['vacancy_title', 'vacancy_description', 'text_clean', 'skills_str',
                  'specialization', 'profile', 'velocity', 'is_top_performer']

    df[final_cols].to_parquet(output_parquet, index=False)
    print(f"✅ Готово! Собрано вакансий: {len(df)}. Эталонов для обучения: {df['is_top_performer'].sum()}")


if __name__ == "__main__":
    prepare_dataset('vacancies_export.csv', 'dataset/vacancies_processed.parquet')
