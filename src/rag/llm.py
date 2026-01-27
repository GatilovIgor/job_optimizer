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
                 n_ctx=4096):

        print(f"📦 Loading model: {filename}...", flush=True)
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False
        )
        print("📦 Model ready!", flush=True)

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        print("      [LLM] Generating...", flush=True)

        # Контекст из базы знаний (RAG)
        ref_text = ""
        if references:
            ref_text = f"REFERENCE EXAMPLE (Use this style):\n{references[0]['html_text'][:800]}\nTitle: {references[0]['title']}"

        issues_list = ", ".join(issues)

        # Входные данные (что заполнил юзер)
        title = user_vacancy.get('title', '')
        text = user_vacancy.get('text', '')
        spec = user_vacancy.get('specialization', '')

        # Формируем задачу для LLM
        system_prompt = (
            "You are an expert HR Recruiter. Your goal is to create a PERFECT vacancy description in Russian.\n"
            "Rules:\n"
            "1. If Title is missing -> Extract it from text OR invent a suitable one.\n"
            "2. If Sphere is missing -> Infer it from title/text.\n"
            "3. If Text is missing -> WRITE IT FROM SCRATCH based on Title.\n"
            "4. If Text exists -> Rewrite it to be structured (HTML), attractive, and detailed.\n"
            "5. Output valid JSON only."
        )

        user_message = (
            f"{ref_text}\n\n"
            f"USER INPUT:\n"
            f"Title: {title if title else '(MISSING - Create one!)'}\n"
            f"Sphere: {spec if spec else '(MISSING - Infer it!)'}\n"
            f"Text: {text if text else '(MISSING - Write a full vacancy description!)'}\n\n"
            f"Fix these issues: {issues_list}\n\n"
            "RESPONSE FORMAT (JSON):\n"
            "{\n"
            "  \"title\": \"Полное название должности\",\n"
            "  \"specialization\": \"Сфера (Продажи, IT...)\",\n"
            "  \"rewritten_text\": \"<p>Описание компании...</p><h3>Обязанности:</h3><ul><li>...</li></ul>...\",\n"
            "  \"rewrite_notes\": [\"Придумал заголовок\", \"Добавил структуру\"]\n"
            "}"
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            content = response['choices'][0]['message']['content']
            clean_json = re.sub(r"```json|```", "", content).strip()
            data = json.loads(clean_json)

            # Фоллбэки, если модель что-то забыла вернуть
            return {
                "title": data.get("title", title or "Специалист"),
                "specialization": data.get("specialization", spec or "Общее"),
                "rewritten_text": data.get("rewritten_text", text),
                "rewrite_notes": data.get("rewrite_notes", [])
            }

        except Exception as e:
            print(f"      ❌ LLM Error: {e}", flush=True)
            return {
                "title": title, "specialization": spec,
                "rewritten_text": text if text else "<p>Ошибка генерации</p>",
                "rewrite_notes": ["Ошибка AI"]
            }
