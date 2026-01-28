import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm  # Нужна библиотека tqdm для прогресс-бара

# --- НАСТРОЙКА ПУТЕЙ ---
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

# Переключаемся на БОЛЬШОЙ файл
RAW_FILE = DATA_DIR / "fact_vacancies_raw.csv"
OUTPUT_FILE = DATA_DIR / "vacancies_processed.parquet"

# Колонки, которые нам реально нужны (чтобы экономить память)
# Не грузим лишний мусор типа source, company_original и т.д.
REQUIRED_COLS = [
    'vacancy_id', 'loaded_at', 'total_responses',
    'profile', 'city', 'vacancy_title',
    'vacancy_description', 'specialization'
]


def calculate_peak_efficiency(dates, responses, window_days=7):
    """
    Оптимизированный расчет эффективности на NumPy.
    """
    n = len(dates)
    if n < 2: return 0.0

    # Конвертируем в наносекунды для быстрого сравнения, если это datetime64
    # Но проще работать с индексами, так как массив отсортирован

    best_eff = 0.0

    # Окно в наносекундах (7 дней)
    window_ns = np.timedelta64(window_days, 'D').astype('timedelta64[ns]').astype(np.int64)
    dates_ns = dates.astype(np.int64)

    # Проходим по массиву
    for i in range(n):
        start_time = dates_ns[i]
        limit_time = start_time + window_ns

        # Бинарный поиск конца окна (очень быстро)
        end_idx = np.searchsorted(dates_ns, limit_time, side='right') - 1

        if end_idx <= i: continue

        val_start = responses[i]
        val_end = responses[end_idx]

        current_eff = 0

        if val_start > 0:
            current_eff = val_end - val_start
        else:
            # Логика "первого ненулевого"
            # Срез внутри окна
            window_slice = responses[i: end_idx + 1]
            # np.argmax возвращает индекс первого True (или 0)
            # Проверяем, есть ли вообще значения > 0
            is_nonzero = window_slice > 0
            if is_nonzero.any():
                first_nonzero_idx = np.argmax(is_nonzero)
                first_val = window_slice[first_nonzero_idx]
                current_eff = val_end - first_val
            else:
                current_eff = 0

        if current_eff > best_eff:
            best_eff = current_eff

    return float(best_eff)


def main():
    print(f"🚀 Рабочая директория: {ROOT_DIR}")
    print(f"📂 Загрузка большого файла: {RAW_FILE.name}...")

    if not RAW_FILE.exists():
        print(f"❌ Файл {RAW_FILE} не найден!")
        return

    # 1. Загружаем только нужные колонки
    try:
        df = pd.read_csv(
            RAW_FILE,
            usecols=lambda c: c in REQUIRED_COLS,  # Грузим только то, что есть в списке
            low_memory=False
        )
    except ValueError as e:
        # Если вдруг названия колонок отличаются (например нет profile), пробуем загрузить всё
        print(f"⚠️ Ошибка фильтрации колонок ({e}), пробуем загрузить всё...")
        df = pd.read_csv(RAW_FILE, low_memory=False)

    print(f"📦 Загружено строк: {len(df)}")

    # Приводим типы
    df['loaded_at'] = pd.to_datetime(df['loaded_at'])
    df['total_responses'] = df['total_responses'].fillna(0).astype(int)
    # Приводим ID к строке, чтобы избежать путаницы int/str
    df['vacancy_id'] = df['vacancy_id'].astype(str)

    print("⏳ Группировка данных по вакансиям...")
    grouped = df.groupby('vacancy_id')
    unique_vacancies = len(grouped)
    print(f"🆔 Уникальных вакансий: {unique_vacancies}")

    print("🧠 Расчет пиковой эффективности (это займет время)...")

    results = []

    # Используем tqdm для отображения прогресс-бара
    for vac_id, group in tqdm(grouped, total=unique_vacancies, unit="vac"):
        # Сортировка обязательна для логики окна
        group = group.sort_values('loaded_at')

        # Извлекаем numpy массивы для скорости
        dates = group['loaded_at'].values
        responses = group['total_responses'].values

        eff = calculate_peak_efficiency(dates, responses)

        # Сохраняем "свежайшую" версию описания
        # (iloc[-1] берет последнюю запись по времени)
        best_row = group.iloc[-1].to_dict()
        best_row['efficiency'] = eff

        # Удаляем лишнее из словаря, чтобы не дублировать память
        # (loaded_at и total_responses нам в RAG уже не нужны, нужна только метрика)
        del best_row['loaded_at']
        del best_row['total_responses']

        results.append(best_row)

    result_df = pd.DataFrame(results)

    # Аналитика по метрике
    max_eff = result_df['efficiency'].max()
    avg_eff = result_df['efficiency'].mean()
    print(f"\n📊 Статистика эффективности:")
    print(f"   Максимум: {max_eff:.1f} откликов/неделю")
    print(f"   Среднее:  {avg_eff:.1f} откликов/неделю")

    # Топ перформеры (Top 20%)
    # Если данных мало или все нули, берем хотя бы > 0
    threshold = result_df['efficiency'].quantile(0.8)
    if threshold == 0 and max_eff > 0:
        print("⚠️ 80-й перцентиль равен 0. Будем считать топами всех, у кого > 0.")
        threshold = 1.0

    result_df['is_top_performer'] = result_df['efficiency'] >= threshold

    top_count = result_df['is_top_performer'].sum()
    print(f"🏆 Порог Top-20%: {threshold:.1f}")
    print(f"🌟 Эталонных вакансий отобрано: {top_count}")

    # Сохранение
    print(f"💾 Сохранение в Parquet...")
    result_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Успешно! Файл готов: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
