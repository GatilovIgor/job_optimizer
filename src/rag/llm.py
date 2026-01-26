import os
import json
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                 filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                 n_ctx=2048):

        print(f"📦 Loading model: {filename}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            n_threads=8,
            verbose=False
        )

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        # Упрощенный контекст
        ref_text = f"STYLE SAMPLE:\n{references[0]['html_text'][:600]}" if references else ""
        issues_list = ", ".join(issues) if issues else "None"

        # Системный промпт
        system_prompt = (
            "You are an expert HR copywriter. Rewrite the vacancy in Russian. "
            "Make it attractive and structured (HTML format). "
            "Output MUST be valid JSON."
        )

        user_message = (
            f"{ref_text}\n\n"
            f"OLD TITLE: {user_vacancy.get('title')}\n"
            f"OLD TEXT: {user_vacancy['text']}\n"
            f"ISSUES: {issues_list}\n\n"
            "TASK: Rewrite the text in Russian. Use HTML (<ul>, <li>, <strong>).\n"
            "RETURN JSON:\n"
            "{\n"
            "  \"rewritten_text\": \"<p>Здесь новый текст...</p>\",\n"
            "  \"rewrite_notes\": [\"Исправил тон\", \"Добавил списки\"]\n"
            "}"
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"}
        )

        content = response['choices'][0]['message']['content']

        # Надежный парсинг
        try:
            # Убираем markdown обертку, если модель ее добавила
            clean_json = re.sub(r"```json|```", "", content).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            print(f"❌ JSON Error. Raw content: {content}")
            # Возвращаем аварийный ответ, чтобы интерфейс не падал
            return {
                "rewritten_text": f"<p>Не удалось сгенерировать текст. Сырой ответ нейросети:</p><pre>{content}</pre>",
                "rewrite_notes": ["Ошибка формата JSON"]
            }
