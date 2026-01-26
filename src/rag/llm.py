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
                 n_ctx=4096):

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,
            verbose=False
        )

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        ref_text = ""
        for i, r in enumerate(references):
            ref_text += f"\n--- {i + 1} ---\n{r['title']}\n"

        issues_text = "\n".join([f"- {i}" for i in issues]) if issues else ""

        system_prompt = (
            "Ты — профессиональный HR-редактор. "
            "Твоя задача — переписать текст вакансии, опираясь на успешные примеры.\n"
            "1. Не выдумывай факты.\n"
            "2. Ответ строго в JSON.\n"
        )

        user_message = (
            f"Title: {user_vacancy.get('title', '')}\nText: {user_vacancy['text']}\n\n"
            f"ISSUES:\n{issues_text}\n\n"
            f"REFS:{ref_text}\n\n"
            "JSON format:\n"
            "{\n"
            '  "rewritten_text": "text",\n'
            '  "rewrite_notes": ["note1"],\n'
            '  "safety_flags": []\n'
            "}"
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        content = response['choices'][0]['message']['content']

        try:
            clean_json = re.sub(r"```json|```", "", content).strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            return {
                "rewritten_text": user_vacancy['text'],
                "rewrite_notes": ["JSON Error"],
                "safety_flags": ["JSON_PARSING_ERROR"]
            }
