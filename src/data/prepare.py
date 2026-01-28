import pandas as pd
import numpy as np
import pathlib

# --- НАСТРОЙКА ПУТЕЙ ---
# Файл лежит в src/data/prepare.py
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
# Поднимаемся на 2 уровня: src/data -> src -> root
ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Путь к исходному CSV и итоговому Parquet
RAW_FILE = DATA_DIR / "fact_vacancies_test.csv"
OUTPUT_FILE = DATA_DIR / "vacancies_processed.parquet"


def calculate_peak_efficiency(group, window_days=7):
    """
    Рассчитывает метрику: Максимальный прирост откликов за 7 дней.
    """
    group = group.sort_values('loaded_at')
    dates = group['loaded_at'].values
    responses = group['total_responses'].fillna(0).values

    n = len(group)
    if n < 2: return 0.0

    best_eff = 0.0

    for i in range(n):
        start_date = dates[i]
        limit_date = start_date + np.timedelta64(window_days, 'D')

        # Индекс конца окна
        end_idx = np.searchsorted(dates, limit_date, side='right') - 1

        if end_idx <= i: continue

        val_end = responses[end_idx]
        val_start = responses[i]

        current_eff = 0
        if val_start > 0:
            current_eff = val_end - val_start
        else:
            # Если начало 0, ищем первый >0
            window_slice = responses[i: end_idx + 1]
            nonzero_indices = np.nonzero(window_slice)[0]
            if len(nonzero_indices) > 0:
                first_nonzero = window_slice[nonzero_indices[0]]
                current_eff = val_end - first_nonzero
            else:
                current_eff = 0

        if current_eff > best_eff:
            best_eff = current_eff

    return float(best_eff)


def main():
    print(f"🚀 Рабочая директория: {ROOT_DIR}")
    print(f"📂 Ищем файл данных: {RAW_FILE}")

    if not RAW_FILE.exists():
        print(f"❌ ОШИБКА: Файл не найден! Положите 'fact_vacancies_test.csv' в папку 'data/'.")
        return

    df = pd.read_csv(RAW_FILE)
    df['loaded_at'] = pd.to_datetime(df['loaded_at'])

    print(f"📦 Загружено строк: {len(df)}")

    # Расчет метрики
    print("⏳ Расчет пиковой эффективности (может занять время)...")
    best_vacancies = []

    for vac_id, group in df.groupby('vacancy_id'):
        eff = calculate_peak_efficiency(group)
        # Берем последнюю версию описания
        best_row = group.sort_values('loaded_at').iloc[-1].to_dict()
        best_row['efficiency'] = eff
        best_vacancies.append(best_row)

    result_df = pd.DataFrame(best_vacancies)

    # Топ перформеры (Top 20%)
    threshold = result_df['efficiency'].quantile(0.8)
    result_df['is_top_performer'] = result_df['efficiency'] >= threshold

    print(f"📊 Порог (Top 20%): {threshold:.2f} откликов/неделю")

    # Сохранение
    cols = [
        'vacancy_id', 'profile', 'city', 'vacancy_title',
        'vacancy_description', 'specialization',
        'efficiency', 'is_top_performer'
    ]
    # Оставляем только те, что есть в df
    save_cols = [c for c in cols if c in result_df.columns]

    result_df[save_cols].to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Готово! Файл сохранен: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
