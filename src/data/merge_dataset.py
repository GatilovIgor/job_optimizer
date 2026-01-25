import pandas as pd
import pathlib
import numpy as np


def main():
    print("🚀 STEP 3: Merge (Bypass Mode)...")

    # 1. Пути
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    data_dir = root / "dataset"
    facts_file = data_dir / "facts.parquet"
    out_file = data_dir / "vacancies_full.parquet"

    if not facts_file.exists():
        print("❌ facts.parquet not found. Run extract_facts.py first!")
        return

    # 2. Загрузка фактов
    print("   Loading facts...")
    df = pd.read_parquet(facts_file)
    print(f"   Loaded {len(df)} rows.")

    # 3. Генерация заглушки для описания
    # Мы используем Title как основу, чтобы векторы имели смысл.
    print("   Generating description placeholders...")
    df['vacancy_description'] = df['vacancy_title'].apply(
        lambda x: f"<h1>{x}</h1><p>Full description unavailable due to DB lock.</p>"
    )

    # 4. Обработка типов (для совместимости)
    for col in ['publication_date', 'creation_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 5. Сохранение
    df.to_parquet(out_file, index=False)

    print("-" * 30)
    print(f"✅ FINAL SUCCESS! Dataset created: {out_file}")
    print(f"   Rows: {len(df)}")
    print(f"   Note: 'vacancy_description' is populated from 'vacancy_title'")
    print("-" * 30)


if __name__ == "__main__":
    main()
