import pandas as pd
import pathlib
import numpy as np


def main():
    print("🚀 STEP 3: Merge (Real Data Mode)...")

    # 1. Пути
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    facts_file = data_dir / "facts.parquet"
    desc_file = data_dir / "descriptions.parquet"
    out_file = data_dir / "vacancies_full.parquet"

    if not facts_file.exists():
        print(f"❌ facts.parquet not found in {data_dir}. Run extract_facts.py first!")
        return

    # 2. Загрузка фактов
    print("   Loading facts...")
    df_facts = pd.read_parquet(facts_file)
    print(f"   Facts rows: {len(df_facts)}")

    # 3. Загрузка и слияние описаний
    if desc_file.exists():
        print("   Loading descriptions...")
        df_desc = pd.read_parquet(desc_file)
        print(f"   Descriptions rows: {len(df_desc)}")

        # Merge: объединяем факты и описания по ID
        df = pd.merge(df_facts, df_desc, on='vacancy_id', how='left')

        # Если для какой-то вакансии нет описания, пишем текст-заполнитель
        df['vacancy_description'] = df['vacancy_description'].fillna("Description unavailable")
        print("   ✅ Merged facts with descriptions.")
    else:
        print(f"⚠️ Warning: descriptions.parquet not found in {data_dir}!")
        print("   Falling back to empty descriptions.")
        df = df_facts.copy()
        df['vacancy_description'] = ""

    # 4. Обработка дат
    for col in ['publication_date', 'creation_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 5. Сохранение
    df.to_parquet(out_file, index=False)

    print("-" * 30)
    print(f"✅ FINAL SUCCESS! Dataset created: {out_file}")
    print(f"   Total Rows: {len(df)}")
    print("-" * 30)


if __name__ == "__main__":
    main()
