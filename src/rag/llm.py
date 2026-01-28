import os
import re
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                 filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                 n_ctx=2048):

        if Llama is None:
            raise ImportError("Please install llama-cpp-python")

        print(f"📦 checking model: {filename}...", flush=True)
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)

            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                n_threads=6,
                verbose=False
            )
            print("📦 Model loaded!", flush=True)
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            self.llm = None

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        if not self.llm:
            return {"raw_response": "Ошибка: Модель не загружена"}

        print("      [LLM] Generating (Text Mode)...", flush=True)

        # Контекст
        ref_text = ""
        if references:
            ref_content = references[0].get('html_text', '')[:800]
            ref_text = f"ПРИМЕР (СТИЛЬ):\n{ref_content}..."

        issues_str = ", ".join(issues) if issues else "Улучши структуру и продающий стиль."
        title = user_vacancy.get('title', 'Сотрудник')
        text = user_vacancy.get('text', '')

        if len(text) < 100:
            text += "\n(Это черновик. Придумай полноценное описание с обязанностями, требованиями и условиями.)"

        # ПРОМПТ БЕЗ JSON (Просим просто текст)
        system_prompt = (
            "Ты — опытный HR-редактор. Напиши ЛУЧШЕЕ описание вакансии на русском языке.\n"
            "Формат ответа СТРОГО такой:\n"
            "ЗАГОЛОВОК: [Название должности]\n"
            "СФЕРА: [Сфера деятельности]\n"
            "ОПИСАНИЕ:\n"
            "[Вступление]\n"
            "<h3>Обязанности:</h3>\n<ul><li>...</li></ul>\n"
            "<h3>Требования:</h3>\n<ul><li>...</li></ul>\n"
            "<h3>Условия:</h3>\n<ul><li>...</li></ul>"
        )

        user_message = (
            f"{ref_text}\n\n"
            f"ЗАДАЧА: Исправь и дополни вакансию.\n"
            f"Текущая должность: {title}\n"
            f"Исходный текст: {text}\n"
            f"Что исправить: {issues_str}\n\n"
            "Начинай ответ сразу с поля 'ЗАГОЛОВОК:'."
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.6,  # Больше креатива
                max_tokens=1600,
            )

            raw_text = response['choices'][0]['message']['content']
            return {"raw_response": raw_text}

        except Exception as e:
            print(f"      ❌ LLM Gen Error: {e}", flush=True)
            return {"raw_response": text}
