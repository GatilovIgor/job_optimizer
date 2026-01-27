import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import pathlib

# Настройка путей
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Патчим Advisor до импорта app
with patch("src.rag.advisor.VacancyAdvisor") as MockAdvisor:
    from src.api.main import app

client = TestClient(app)

# --- 7 СЦЕНАРИЕВ ЗАПОЛНЕНИЯ ---
COMBINATION_SCENARIOS = [
    # 1. Только Название
    {
        "input_id": "case_1_title_only",
        "title": "Python Developer",
        "text": "",
        "specialization": ""
    },
    # 2. Только Описание
    {
        "input_id": "case_2_text_only",
        "title": "",
        "text": "Нужно писать код на Python и Django, удаленка.",
        "specialization": ""
    },
    # 3. Только Специальность
    {
        "input_id": "case_3_spec_only",
        "title": "",
        "text": "",
        "specialization": "IT / Backend Development"
    },
    # 4. Название + Описание
    {
        "input_id": "case_4_title_text",
        "title": "Python Dev",
        "text": "Работа в офисе, полный день.",
        "specialization": ""
    },
    # 5. Название + Специальность
    {
        "input_id": "case_5_title_spec",
        "title": "Team Lead",
        "text": "",
        "specialization": "Management"
    },
    # 6. Описание + Специальность
    {
        "input_id": "case_6_text_spec",
        "title": "",
        "text": "Руководство командой разработки из 5 человек.",
        "specialization": "IT Management"
    },
    # 7. ВСЕ ТРИ ПОЛЯ (Happy Path)
    {
        "input_id": "case_7_full",
        "title": "Senior Python Developer",
        "text": "Highload проекты, микросервисы.",
        "specialization": "IT"
    }
]


# Умная заглушка, которая показывает, что она "видит"
def check_what_is_filled(vacancy, retriever):
    # Логика заглушки: формируем ответ на основе того, что пришло
    filled_fields = []
    if vacancy.title: filled_fields.append("TITLE")
    if vacancy.text: filled_fields.append("TEXT")
    if vacancy.specialization: filled_fields.append("SPEC")

    context_str = "+".join(filled_fields)

    return {
        "input_id": vacancy.input_id,
        # В ответе мы явно пишем, какие поля были использованы для генерации
        "rewritten_title": f"[Generated from {context_str}] {vacancy.title or 'New Title'}",
        "rewritten_text": f"Based on {context_str}: {vacancy.text or 'Generated description...'}",
        "rewritten_specialization": vacancy.specialization or "Detected Spec",
        "quality_score": 50 + (len(filled_fields) * 15),  # Чем больше полей, тем выше балл
        "original_score": 10,
        "issues": [],
        "rewrite_notes": [f"Used fields: {context_str}"],
        "safety_flags": [],
        "low_confidence_retrieval": False
    }


def test_field_combinations():
    """
    Проверяет, как система реагирует на разные комбинации заполненных полей.
    """
    payload = {"vacancies": COMBINATION_SCENARIOS}

    with patch("src.api.main.advisor") as mock_adv:
        mock_adv.process_single_vacancy.side_effect = check_what_is_filled

        print(f"\n🧪 Тестируем {len(COMBINATION_SCENARIOS)} комбинаций полей...")

        response = client.post("/rewrite-batch", json=payload)

        assert response.status_code == 200
        results = response.json()["results"]

        # Проверяем каждый кейс
        for i, res in enumerate(results):
            case_id = res["input_id"]
            notes = res["rewrite_notes"][0]
            score = res["quality_score"]

            print(f"  ✅ {case_id}: {notes} (Score: {score})")

            # Проверка логики (просто убедимся, что заглушка отработала уникально для каждого)
            if "case_7_full" in case_id:
                assert score >= 90  # Максимальный балл за все поля
            if "case_1" in case_id:
                assert score < 70  # Меньше балл, так как только 1 поле

