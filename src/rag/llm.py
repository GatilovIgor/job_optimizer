import os
import json
import re
import sys
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
        # Убрали n_threads=8, пусть Llama сама решит или использует по умолчанию (обычно стабильнее)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False
        )
        print("📦 Model loaded successfully!", flush=True)

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        print("      [LLM] Preparing prompt...", flush=True)
        ref_text = f"STYLE SAMPLE:\n{references[0]['html_text'][:600]}" if references else ""
        issues_list = ", ".join(issues) if issues else "None"

        title = user_vacancy.get('title', '')
        text = user_vacancy.get('text', '')
        spec = user_vacancy.get('specialization', '')

        input_info = []
        if title: input_info.append(f"CURRENT TITLE: {title}")
        if spec: input_info.append(f"CURRENT SPHERE: {spec}")
        if text:
            input_info.append(f"CURRENT TEXT: {text}")
        else:
            input_info.append("CURRENT TEXT: (Missing, generate based on Title/Sphere)")

        input_block = "\n".join(input_info)

        system_prompt = (
            "You are an expert HR copywriter. Structure and rewrite the vacancy.\n"
            "1. Define Title (Russian).\n"
            "2. Define Sphere (Russian).\n"
            "3. Rewrite Text (Russian, HTML).\n"
            "Output JSON."
        )

        user_message = (
            f"{ref_text}\n\n"
            f"{input_block}\n"
            f"ISSUES: {issues_list}\n\n"
            "TASK: Create a perfect vacancy.\n"
            "RETURN JSON:\n"
            "{\n"
            "  \"title\": \"...\",\n"
            "  \"specialization\": \"...\",\n"
            "  \"rewritten_text\": \"...\",\n"
            "  \"rewrite_notes\": [\"...\"]\n"
            "}"
        )

        print("      [LLM] Generating tokens... (Check your CPU usage)", flush=True)
        try:
            # Уменьшил токены до 1000 для скорости теста
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            print("      [LLM] Generation complete!", flush=True)
        except Exception as e:
            print(f"      ❌ [LLM] CRASH during generation: {e}", flush=True)
            return {}

        content = response['choices'][0]['message']['content']

        try:
            clean_json = re.sub(r"```json|```", "", content).strip()
            data = json.loads(clean_json)
            return {
                "title": data.get("title", title),
                "specialization": data.get("specialization", spec),
                "rewritten_text": data.get("rewritten_text", ""),
                "rewrite_notes": data.get("rewrite_notes", [])
            }
        except json.JSONDecodeError:
            print(f"      ⚠️ [LLM] JSON Error. Raw: {content[:100]}...", flush=True)
            return {
                "title": title,
                "specialization": spec,
                "rewritten_text": content,  # Возвращаем хотя бы текст
                "rewrite_notes": ["JSON Error"]
            }
