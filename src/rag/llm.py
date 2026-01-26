import os
import json
import re
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                 filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                 n_ctx=2048):

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            n_threads=16,
            verbose=False
        )

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        ref_text = ""
        if references:
            r = references[0]
            ref_text = f"ПРИМЕР СТИЛЯ:\n{r['html_text'][:800]}"

        issues_text = ", ".join(issues) if issues else "проблем нет"

        # ЖЕСТКАЯ УСТАНОВКА НА РУССКИЙ ЯЗЫК
        system_prompt = (
            "Ты — профессиональный HR-редактор. Твоя задача — переписать вакансию на РУССКОМ языке. "
            "Используй HTML-разметку. Текст должен быть продающим и структурированным. "
            "Отвечай строго в формате JSON."
        )

        user_message = (
            f"{ref_text}\n\n"
            f"ИСХОДНЫЙ ЗАГОЛОВОК: {user_vacancy.get('title')}\n"
            f"ИСХОДНЫЙ ТЕКСТ: {user_vacancy['text']}\n"
            f"ЧТО ИСПРАВИТЬ: {issues_text}\n\n"
            "ИНСТРУКЦИЯ:\n"
            "1. Пиши только на русском языке.\n"
            "2. Исправь все ошибки и добавь структуру (списки).\n"
            "3. Верни JSON: {\"rewritten_text\": \"...\", \"rewrite_notes\": [\"что исправлено на русском\"]}"
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        content = response['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except:
            return {
                "rewritten_text": user_vacancy['text'],
                "rewrite_notes": ["Ошибка генерации JSON. Попробуйте еще раз."]
            }
