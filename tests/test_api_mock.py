from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import pathlib

# Добавляем корень проекта в sys.path, чтобы видеть модули
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Импортируем app, но перед этим нужно замокать advisor,
# чтобы при импорте main.py не началась загрузка модели
with patch("src.rag.advisor.VacancyAdvisor") as MockAdvisor:
    from src.api.main import app

client = TestClient(app)


def test_api_rewrite_batch():
    # 1. Подготовка заглушки (Mock)
    # Добавляем недостающие поля rewritten_title и rewritten_specialization
    mock_result_data = {
        "input_id": "test_1",
        "rewritten_title": "Mocked Title",  # <--- ДОБАВЛЕНО
        "rewritten_text": "Mocked Text",
        "rewritten_specialization": "Mocked Specialization", # <--- ДОБАВЛЕНО
        "quality_score": 100,
        "original_score": 50,
        "issues": [],
        "rewrite_notes": [],
        "safety_flags": [],
        "low_confidence_retrieval": False
    }

    # Внедряем заглушку в глобальную переменную advisor внутри main.py
    with patch("src.api.main.advisor") as mock_adv:
        # Теперь process_single_vacancy будет возвращать полный словарь
        mock_adv.process_single_vacancy.return_value = mock_result_data

        # 2. Сам запрос
        payload = {
            "vacancies": [{
                "input_id": "test_1",
                "title": "Test",
                "text": "Short text"
            }]
        }

        response = client.post("/rewrite-batch", json=payload)

        # 3. Проверки
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["rewritten_text"] == "Mocked Text"
        assert data["results"][0]["rewritten_title"] == "Mocked Title" # <--- ДОБАВЛЕНА ПРОВЕРКА
        assert data["results"][0]["input_id"] == "test_1"

