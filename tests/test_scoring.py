import pytest
from src.rag.advisor import VacancyAdvisor


@pytest.fixture
def advisor():
    original_init = VacancyAdvisor.__init__
    VacancyAdvisor.__init__ = lambda self: None
    adv = VacancyAdvisor()
    VacancyAdvisor.__init__ = original_init
    return adv


def test_score_empty_text(advisor):
    res = advisor._analyze_quality("")
    assert res["score"] == 0
    # Проверка на текст < 50 символов
    assert "Текст отсутствует" in res["issues"][0]


def test_score_poor_vacancy(advisor):
    """Вакансия-огрызок (но длиннее 50 символов)"""
    # Добавили мусора, чтобы длина стала > 50
    text = "Требуется продавец. График 5/2. Зп хорошая. Очень нужно найти человека срочно."
    res = advisor._analyze_quality(text)

    assert res["score"] < 40
    issues_str = " ".join(res["issues"])
    # Теперь он должен попасть в ветку 'Критически мало текста'
    assert "Критически мало текста" in issues_str
    assert "Нет блока 'Обязанности'" in issues_str


def test_score_rich_vacancy(advisor):
    text = """
    Мы ищем Менеджера по продажам в офис.

    Обязанности:
    • Звонить клиентам
    • Вести базу

    Требования:
    • Опыт работы
    • Активность

    Условия:
    • Оклад 50000 руб + премия
    • График 5/2, оформление по ТК РФ
    """
    res = advisor._analyze_quality(text)

    assert res["score"] > 60
    assert not any("Обязанности" in i for i in res["issues"])
