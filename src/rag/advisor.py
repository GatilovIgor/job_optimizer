from typing import List, Dict
from src.rag.llm import LocalLLM


class VacancyAdvisor:
    def __init__(self):
        # Инициализируем локальную LLM
        # Она сама скачает модель при первом запуске
        self.llm = LocalLLM()

    def analyze(self, user_vacancy: str, references: List[Dict]) -> Dict:
        """
        Принимает текст вакансии пользователя и список успешных примеров.
        Генерирует советы с помощью LLM.
        """
        if not references:
            return {
                "ai_advice_text": "К сожалению, мы не нашли похожих успешных вакансий для сравнения. Попробуйте изменить название."
            }

        print("🤖 AI is generating advice...")

        # Запускаем генерацию
        ai_recommendation = self.llm.generate_advice(user_vacancy, references)

        return {
            "ai_advice_text": ai_recommendation
        }


# --- ТЕСТ ---
if __name__ == "__main__":
    advisor = VacancyAdvisor()
    refs = [{"title": "Senior Python Developer"}, {"title": "Python Team Lead"}]
    res = advisor.analyze("Ищем питониста", refs)
    print(res['ai_advice_text'])
