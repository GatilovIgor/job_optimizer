import os
import json
from huggingface_hub import InferenceClient, get_token
from src.api.models import VacancyOut, VacancyIn


class VacancyOptimizer:
    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or get_token()

        if not token:
            print("⚠️ HUGGINGFACE TOKEN НЕ НАЙДЕН. Убедитесь, что настроили .env или сделали hf login")

        # Инициализируем клиент напрямую
        # Qwen/Qwen2.5-72B-Instruct - это мощнейшая модель, она требует chat-интерфейса
        self.client = InferenceClient(
            model="Qwen/Qwen2.5-72B-Instruct",
            token=token,
            timeout=120
        )

    def optimize(self, vac: VacancyIn, references: list) -> VacancyOut:
        # 1. Подготовка контекста (референсов)
        refs_text = ""
        for i, r in enumerate(references[:2]):
            title = r.get('vacancy_title', 'Без заголовка')
            # Обрезаем описание, чтобы не забить контекст
            desc = str(r.get('vacancy_description', '')).replace('\n', ' ')[:300]
            refs_text += f"- Пример {i + 1}: {title} | {desc}...\n"

        # 2. Формируем сообщения для чата (System + User)
        system_message = """You are a professional HR Expert and Copywriter. 
Your goal is to rewrite job descriptions to maximize applicant conversion.
You MUST reply with valid JSON only. No markdown, no conversational filler."""

        user_content = f"""
I need you to optimize this vacancy based on successful examples.

INPUT DATA:
Profile: {vac.profile}
City: {vac.city}
Title: {vac.vacancy_title}
Specialization: {vac.specialization}
Description: {vac.vacancy_description}

SUCCESSFUL EXAMPLES (Use style and structure from here):
{refs_text}

INSTRUCTIONS:
1.  **Title**: Make it specific and attractive.
2.  **Structure**: Organize Description into clear sections: "Обязанности" (Responsibilities), "Требования" (Requirements), "Условия" (Conditions).
3.  **Tone**: Professional but inviting.
4.  **Output**: Return strictly valid JSON.

JSON SCHEMA:
{{
    "vacancy_title": "New Title",
    "vacancy_description": "New formatted description...",
    "profile": "Confirmed Profile",
    "city": "Confirmed City",
    "specialization": "Confirmed Specialization",
    "improvement_notes": ["Note 1", "Note 2"]
}}
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        try:
            # 3. Отправляем запрос как ЧАТ (chat_completion)
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=2500,
                temperature=0.2,  # Низкая температура для строгости JSON
                top_p=0.9
            )

            # Получаем текст ответа
            raw_content = response.choices[0].message.content

            # 4. Чистка JSON (Qwen любит оборачивать в ```json ... ```)
            clean_json = raw_content.strip()

            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0]
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0]

            # Убираем лишний текст до и после JSON
            start_idx = clean_json.find("{")
            end_idx = clean_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_json[start_idx: end_idx + 1]

            # 5. Парсинг
            data = json.loads(clean_json)

            return VacancyOut(
                input_id=vac.input_id,
                profile=data.get("profile", vac.profile),
                city=data.get("city", vac.city),
                vacancy_title=data.get("vacancy_title", vac.vacancy_title),
                specialization=data.get("specialization", vac.specialization),
                vacancy_description=data.get("vacancy_description", vac.vacancy_description),
                improvement_notes=data.get("improvement_notes", ["Оптимизация структуры и стиля"]),
                predicted_efficiency_score=None
            )

        except Exception as e:
            print(f"❌ Ошибка Qwen API: {e}")
            # Если сломалось - возвращаем исходник с ошибкой
            return VacancyOut(
                input_id=vac.input_id,
                profile=vac.profile,
                city=vac.city,
                vacancy_title=vac.vacancy_title,
                specialization=vac.specialization,
                vacancy_description=vac.vacancy_description,
                improvement_notes=[f"API Error: {str(e)}"],
                predicted_efficiency_score=None
            )
