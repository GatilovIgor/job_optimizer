FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_NO_CACHE_DIR=1

# Ставим корректные системные либы
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential gcc libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем список зависимостей
COPY requirements.txt .

# Устанавливаем всё (строго CPU версии)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем код
COPY . .

RUN mkdir -p /app/data /root/.cache/huggingface

EXPOSE 8000
EXPOSE 8501
