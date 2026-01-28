import os
import json
import re
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
except ImportError:
    print("⚠️ llama_cpp not installed. LLM will not work.")
    Llama = None

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                 filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                 n_ctx=4096):

        if Llama is None:
            raise ImportError("Please install llama-cpp-python")

        print(f"📦 checking model: {filename}...", flush=True)
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"   path: {model_path}")

            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=0,  # Ставим 0 для CPU, увеличьте если есть GPU
                verbose=False
            )
            print("📦 Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            self.llm = None

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        if not self.llm:
            return {
                "title": user_vacancy.get('title'),
                "rewritten_text": user_vacancy.get('text') + "\n<p>(LLM not loaded)</p>",
                "rewrite_notes": ["LLM Error"]
            }

        print("      [LLM] Generating...", flush=True)

        # Контекст из базы знаний (RAG)
        ref_text = ""
        if references:
            ref_content = references[0].get('html_text', '')[:800]
            ref_title = references[0].get('title', '')
            ref_text = f"REFERENCE EXAMPLE (Best Practice):\nTitle: {ref_title}\n{ref_content}"

        issues_list = ", ".join(issues)

        title = user_vacancy.get('title', '')
        text = user_vacancy.get('text', '')
        spec = user_vacancy.get('specialization', '')

        system_prompt = (
            "You are an expert HR Recruiter. Your goal is to create a PERFECT vacancy description in Russian.\n"
            "Rules:\n"
            "1. If Title is missing -> Create one.\n"
            "2. If Text is missing -> Write full description.\n"
            "3. Use HTML tags (<h3>, <ul>, <li>, <p>).\n"
            "4. Output valid JSON only."
        )

        user_message = (
            f"{ref_text}\n\n"
            f"USER INPUT:\n"
            f"Title: {title if title else '(MISSING)'}\n"
            f"Sphere: {spec if spec else '(MISSING)'}\n"
            f"Text: {text if text else '(MISSING)'}\n\n"
            f"Fix issues: {issues_list}\n\n"
            "RESPONSE FORMAT (JSON):\n"
            "{\n"
            "  \"title\": \"Str\",\n"
            "  \"specialization\": \"Str\",\n"
            "  \"rewritten_text\": \"HTML String\",\n"
            "  \"rewrite_notes\": [\"Str\"]\n"
            "}"
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response['choices'][0]['message']['content']
            # Чистка JSON от markdown
            clean_json = re.sub(r"```json|```", "", content).strip()
            data = json.loads(clean_json)

            return {
                "title": data.get("title", title or "Специалист"),
                "specialization": data.get("specialization", spec or "Общее"),
                "rewritten_text": data.get("rewritten_text", text),
                "rewrite_notes": data.get("rewrite_notes", [])
            }

        except Exception as e:
            print(f"      ❌ LLM Gen Error: {e}", flush=True)
            return {
                "title": title, "specialization": spec,
                "rewritten_text": text if text else "<p>Ошибка генерации</p>",
                "rewrite_notes": ["Ошибка AI"]
            }
