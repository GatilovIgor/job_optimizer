import pytest
from src.rag.advisor import VacancyAdvisor


@pytest.fixture
def advisor():
    # Инициализируем без загрузки тяжелой LLM, если это возможно,
    # но так как она в __init__, мы просто протестируем метод очистки,
    # который не зависит от LLM.
    # Для чистоты эксперимента лучше замокать LLM, но пока проверим метод напрямую.

    # Хакаем инициализацию, чтобы не грузить модель для простых тестов
    original_init = VacancyAdvisor.__init__
    VacancyAdvisor.__init__ = lambda self: None
    advisor = VacancyAdvisor()
    VacancyAdvisor.__init__ = original_init
    return advisor


def test_clean_html_bristol_case(advisor):
    """Тест на реальном примере с 'Бристоль'"""
    dirty_input = """&lt;p&gt;&lt;strong&gt;«Бристоль»&lt;/strong&gt;&lt;/p&gt; &lt;ul&gt; &lt;li&gt;Опыт&lt;/li&gt; &lt;/ul&gt;"""

    expected = "«Бристоль»\n\n\n• Опыт"
    result = advisor._clean_html(dirty_input)

    # Проверяем, что теги ушли, а суть осталась
    assert "«Бристоль»" in result
    assert "<li>" not in result
    assert "&lt;" not in result


def test_clean_simple_text(advisor):
    """Проверка, что обычный текст не ломается"""
    text = "Привет, мир!"
    assert advisor._clean_html(text) == "Привет, мир!"


def test_clean_structure(advisor):
    """Проверка сохранения структуры списков"""
    html_list = "<ul><li>Раз</li><li>Два</li></ul>"
    cleaned = advisor._clean_html(html_list)
    assert "• Раз" in cleaned
    assert "• Два" in cleaned
