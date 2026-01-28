import os
import json
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from huggingface_hub import get_token
from src.api.models import VacancyOut


class OptimizerLLM:
    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()
        self.llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct",
            task="text-generation",
            max_new_tokens=2000,
            temperature=0.3,  # Пониже для стабильности JSON
            huggingfacehub_api_token=token
        )

    def optimize(self, input_data: dict, references: list) -> VacancyOut:
        # Формируем контекст из примеров
        refs_text = ""
        for i, ref in enumerate(references[:2]):
            refs_text += f"\n--- ПРИМЕР ЭФФЕКТИВНОЙ ВАКАНСИИ {i + 1} ---\n"
            refs_text += f"Title: {ref.get('vacancy_title')}\n"
            refs_text += f"Description Snippet: {ref.get('vacancy_description')[:300]}...\n"

        prompt = f"""
        Ты — профессиональный HR-оптимизатор. Твоя задача — переписать вакансию так, чтобы максимизировать отклики, используя лучшие практики.

        ИСХОДНЫЕ ДАННЫЕ:
        Profile: {input_data.get('profile')}
        City: {input_data.get('city')}
        Title: {input_data.get('vacancy_title')}
        Specialization: {input_data.get('specialization')}
        Description: {input_data.get('vacancy_description')}

        РЕФЕРЕНСЫ (Успешные примеры):
        {refs_text}

        ЗАДАЧА:
        1. Улучши заголовок (сделай продающим).
        2. Структурируй описание (обязанности, требования, условия).
        3. Сохрани город и профиль, если они корректны, или уточни их.

        ВЕРНИ ОТВЕТ СТРОГО В JSON ФОРМАТЕ:
        {{
            "profile": "...",
            "city": "...",
            "vacancy_title": "...",
            "specialization": "...",
            "vacancy_description": "...",
            "improvement_notes": ["что улучшено 1", "что улучшено 2"]
        }}
        """

        try:
            response = self.llm.invoke(prompt)
            # Чистим ответ от markdown ```json ... ```
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            return VacancyOut(
                input_id=input_data.get("input_id", "unknown"),
                profile=data.get("profile", ""),
                city=data.get("city", ""),
                vacancy_title=data.get("vacancy_title", ""),
                specialization=data.get("specialization", ""),
                vacancy_description=data.get("vacancy_description", ""),
                improvement_notes=data.get("improvement_notes", []),
                efficiency_prediction=None  # Пока заглушка
            )
        except Exception as e:
            print(f"LLM Error: {e}")
            # Возвращаем исходные данные в случае ошибки
            return VacancyOut(
                input_id=input_data.get("input_id"),
                profile=input_data.get("profile"),
                city=input_data.get("city"),
                vacancy_title=input_data.get("vacancy_title"),
                specialization=input_data.get("specialization"),
                vacancy_description=input_data.get("vacancy_description"),
                improvement_notes=["Ошибка генерации, возвращен исходник"]
            )
