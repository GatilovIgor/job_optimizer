import os
import json
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                 filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                 n_ctx=8192): # Увеличил контекст до 8k, т.к. HTML объемный

        # Качаем модель, если нет
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0, # Поставьте -1 или число слоев, если есть GPU
            verbose=False
        )

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        # 1. Формируем блок примеров (References)
        ref_text = ""
        for i, r in enumerate(references):
            # Мы показываем LLM исходный HTML успешной вакансии.
            # Это учит её структуре (списки, жирный шрифт).
            description_snippet = r['html_text'][:2000] # Берем первые 2000 символов примера
            ref_text += (
                f"\n--- EXAMPLE {i + 1} (Score: {r['score']:.2f}) ---\n"
                f"Title: {r['title']}\n"
                f"Content:\n{description_snippet}\n"
            )

        # 2. Формируем список проблем
        issues_text = "\n".join([f"- {i}" for i in issues]) if issues else "None identified."

        # 3. Системный промпт
        system_prompt = (
            "You are an expert HR Editor using HTML.\n"
            "Your task is to rewrite the user's vacancy to make it attractive and effective.\n"
            "Follow the style and HTML structure (<ul>, <li>, <strong>) of the provided EXAMPLES.\n"
            "Output must be valid JSON."
        )

        # 4. Сообщение пользователя
        user_message = (
            f"MY VACANCY TITLE: {user_vacancy.get('title', '')}\n"
            f"MY VACANCY TEXT (Draft):\n{user_vacancy['text']}\n\n"
            f"ISSUES TO FIX:\n{issues_text}\n\n"
            f"SUCCESSFUL EXAMPLES (References):\n{ref_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Rewrite the text improving clarity and tone.\n"
            "2. Use HTML tags for lists and emphasis.\n"
            "3. Keep the factual requirements the same.\n"
            "4. Return strict JSON format:\n"
            "{\n"
            '  "rewritten_text": "<p>New html content...</p>",\n'
            '  "rewrite_notes": ["Fixed tone", "Added lists"],\n'
            '  "safety_flags": []\n'
            "}"
        )

        # 5. Генерация
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3, # Низкая температура для стабильного JSON
            max_tokens=2500,
            response_format={"type": "json_object"}
        )

        content = response['choices'][0]['message']['content']

        # 6. Парсинг ответа
        try:
            # Чистим Markdown обертку ```json ... ``` если модель её добавила
            clean_json = re.sub(r"```json|```", "", content).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ JSON Decode Error. Raw content:\n{content}")
            return {
                "rewritten_text": user_vacancy['text'],
                "rewrite_notes": ["Error: AI produced invalid JSON"],
                "safety_flags": ["JSON_PARSING_ERROR"]
            }
