import pandas as pd
import numpy as np
import pathlib
import re
import os
from html.parser import HTMLParser

# --- КОНФИГУРАЦИЯ ПУТЕЙ ---
# Скрипт лежит в src/data/, поднимаемся на 2 уровня вверх к корню проекта
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"


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
    """Удаляет HTML теги из текста."""
    if not isinstance(html_txt, str): return ""
    try:
        s = MLStripper()
        s.feed(html_txt)
        return " ".join(s.get_data().split())
    except:
        return html_txt


def parse_pg_array(array_str):
    """Парсит массив из PostgreSQL вида {1,2,3} в список python."""
    if pd.isna(array_str) or str(array_str) == '{}': return []
    # Удаляем фигурные скобки и разбиваем
    content = str(array_str).strip('{}')
    if not content: return []
    return [int(x) for x in content.split(',') if x.strip().isdigit()]


def clean_skill_name(raw_name):
    """Очищает название навыка от артефактов вида Keyskill': ['..."""
    if pd.isna(raw_name): return ""
    txt = str(raw_name)
    # Убираем начало строки
    txt = txt.replace("Keyskill': ['", "")
    # Убираем возможные хвосты (если есть закрывающие скобки)
    txt = txt.replace("']", "")
    return txt.strip()


def load_skills(skills_path):
    """Загружает и чистит справочник навыков."""
    if not skills_path.exists():
        print(f"⚠️ Файл навыков не найден: {skills_path}")
        return {}

    print(f"📖 Загрузка навыков из {skills_path.name}...")
    df_skills = pd.read_csv(skills_path)

    # Применяем очистку имен
    df_skills['name_clean'] = df_skills['name'].apply(clean_skill_name)

    # Создаем словарь id -> name
    return dict(zip(df_skills['skill_id'], df_skills['name_clean']))


def load_merged_data(data_dir):
    """Ищет CSV файлы с данными (исключая skills.csv) и объединяет их."""
    all_files = list(data_dir.glob("*.csv"))
    # Исключаем skills.csv и файлы экспорта, если они есть, берем только чанки данных
    data_files = [f for f in all_files if f.name != 'skills.csv' and 'vacancies_processed' not in f.name]

    if not data_files:
        print("❌ Не найдено файлов данных (CSV) в папке data!")
        return pd.DataFrame()

    print(f"📦 Найдено файлов данных: {len(data_files)} {[f.name for f in data_files]}")

    dfs = []
    for f in data_files:
        print(f"   + Чтение {f.name}...")
        try:
            # low_memory=False помогает, если типы данных смешаны
            df_chunk = pd.read_csv(f, low_memory=False)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"   ❌ Ошибка чтения {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def main():
    print("🚀 STEP: Data Preparation (Merged & Cleaned)...")

    # 1. Загрузка справочника навыков
    skill_map = load_skills(DATA_DIR / "skills.csv")

    # 2. Загрузка и объединение данных
    df = load_merged_data(DATA_DIR)
    if df.empty:
        print("❌ Нет данных для обработки.")
        return

    print(f"   Всего строк загружено: {len(df)}")

    # 3. Обработка дат
    print("⏳ Расчет времени жизни вакансий...")
    df['upd_date'] = pd.to_datetime(df['last_update_date'], errors='coerce')
    df['pub_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

    # Считаем days_live (разница между обновлением и публикацией)
    # Если даты совпадают или update пустой, ставим минимум 0.1 дня, чтобы не делить на ноль
    df['days_live'] = (df['upd_date'] - df['pub_date']).dt.total_seconds() / (24 * 3600)
    df['days_live'] = df['days_live'].fillna(0).apply(lambda x: max(x, 0.1))

    # 4. Расчет Velocity (отклики в день)
    df['total_responses'] = df['total_responses'].fillna(0)
    df['velocity'] = df['total_responses'] / df['days_live']

    # 5. Определение Top Performers (Эталонов)
    # Берем 80-й перцентиль по скорости набора откликов
    velocity_threshold = df['velocity'].quantile(0.8)

    # Условие: Хорошая скорость И достаточно подробное описание (>300 символов)
    # Обрабатываем описание на случай NaN
    df['vacancy_description'] = df['vacancy_description'].fillna("")

    df['is_top_performer'] = (df['velocity'] >= velocity_threshold) & (df['vacancy_description'].str.len() > 300)

    print(f"   🏆 Порог Velocity (top 20%): {velocity_threshold:.2f}")
    print(f"   🌟 Найдено эталонных вакансий: {df['is_top_performer'].sum()}")

    # 6. Очистка текста и маппинг навыков
    print("🧹 Очистка HTML и маппинг навыков...")
    df['text_clean'] = df['vacancy_description'].apply(strip_tags)

    def map_ids_to_names(ids_str):
        ids = parse_pg_array(ids_str)
        # Берем имя из карты, если нет - пропускаем
        names = [skill_map.get(i) for i in ids if i in skill_map]
        return ", ".join([n for n in names if n])

    if 'skill_ids' in df.columns:
        df['skills_str'] = df['skill_ids'].apply(map_ids_to_names)
    else:
        df['skills_str'] = ""

    # 7. Сохранение
    output_file = DATA_DIR / "vacancies_processed.parquet"
    final_cols = [
        'vacancy_id', 'vacancy_title', 'vacancy_description',
        'text_clean', 'skills_str', 'specialization',
        'profile', 'velocity', 'is_top_performer'
    ]

    # Оставляем только те колонки, которые реально есть в датафрейме
    cols_to_save = [c for c in final_cols if c in df.columns]

    df[cols_to_save].to_parquet(output_file, index=False)
    print(f"✅ Готово! Файл сохранен: {output_file}")


if __name__ == "__main__":
    main()
