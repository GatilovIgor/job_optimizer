import os
import json
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Для Windows отключаем HF Transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                 filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                 n_ctx=4096):
        print(f"⚙️ Initializing Local LLM ({repo_id})...")

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Загружаем (CPU mode)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False
        )

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        """
        Генерирует улучшенную версию вакансии в формате JSON.
        """
        # Формируем контекст референсов
        ref_text = ""
        for i, r in enumerate(references):
            ref_text += f"\n--- УСПЕШНЫЙ ПРИМЕР {i + 1} ---\n{r['title']}\n"

        # Формируем список проблем
        issues_text = "\n".join(
            [f"- {i}" for i in issues]) if issues else "Критических проблем не найдено, просто улучши стиль."

        system_prompt = (
            "Ты — профессиональный HR-редактор и эксперт по найму (Job Optimizer AI). "
            "Твоя задача — переписать текст вакансии, опираясь на успешные примеры, чтобы повысить конверсию откликов.\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            "1. Не выдумывай факты (зарплату, условия, стек), если их нет в исходном тексте.\n"
            "2. Ответ должен быть строго в формате JSON.\n"
            "3. Тон: профессиональный, но привлекательный."
        )

        user_message = (
            f"ИСХОДНАЯ ВАКАНСИЯ:\nTitle: {user_vacancy.get('title', 'Не указан')}\nText: {user_vacancy['text']}\n\n"
            f"НАЙДЕННЫЕ ПРОБЛЕМЫ:\n{issues_text}\n\n"
            f"РЕФЕРЕНСЫ (УСПЕШНЫЕ ВАКАНСИИ):{ref_text}\n\n"
            "Сгенерируй JSON ответ со следующими полями:\n"
            "{\n"
            '  "rewritten_text": "полный переписанный текст вакансии (с HTML форматированием при необходимости)",\n'
            '  "rewrite_notes": ["пункт 1: что улучшено", "пункт 2: почему это важно"],\n'
            '  "safety_flags": ["если пришлось убрать что-то сомнительное" или "если не хватило данных"]\n'
            "}"
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,  # Чуть ниже для стабильности JSON
            max_tokens=1500,
            response_format={"type": "json_object"}  # Подсказка для модели (если поддерживается версией)
        )

        content = response['choices'][0]['message']['content']

        # Попытка парсинга JSON
        try:
            # Иногда модели добавляют Markdown ```json ... ```, чистим
            clean_json = re.sub(r"```json|```", "", content).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error. Raw content: {content[:100]}...")
            return {
                "rewritten_text": user_vacancy['text'],
                "rewrite_notes": ["Ошибка генерации: модель вернула невалидный JSON. Показан исходный текст."],
                "safety_flags": ["JSON_PARSING_ERROR"]
            }
