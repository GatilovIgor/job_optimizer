# job-optimizer-mvp
Инструкция по развертыванию и запуску проекта.

## Предварительные требования
Перед запуском убедитесь, что в корневой директории проекта создан файл `.env` с необходимыми переменными окружения.

---

## Способ 1: Запуск через Docker (Рекомендуемый)

Для автоматической настройки окружения и запуска всех сервисов выполните команду:

```bash
docker-compose up --build


---

## Способ 2: Локальный запуск (Вручную)

Если вы запускаете проект без Docker, выполните следующие шаги последовательно.

### 1. Настройка окружения и установка зависимостей

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Обработка данных (Data Pipeline)

Для корректной работы приложения необходимо подготовить данные. Запустите скрипты в указанном порядке:

**1. Подготовка:**
```bash
python src/data/prepare.py
```

**2. Извлечение данных (Descriptions и Facts):**
```bash
python src/data/extract_facts.py
python src/data/extract_descriptions.py
```

**3. Слияние датасета:**
```bash
python src/data/merge_dataset.py
```

**4. Расчет метрик:**
```bash
python src/data/process_metrics.py
```

### 3. Запуск приложения

После успешного выполнения всех скриптов обработки данных запустите веб-интерфейс:

```bash
python src/demo/app.py
```
*(Примечание: Если проект использует Streamlit, используйте команду `streamlit run src/demo/app.py`)*
```