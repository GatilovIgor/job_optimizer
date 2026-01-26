import os
import pathlib

# Для Windows отключаем HF Transfer, если он не установлен
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class LocalLLM:
    def __init__(self,
                 repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                 filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                 n_ctx=4096):
        print(f"⚙️ Initializing Local LLM ({repo_id})...")

        # 1. Скачиваем модель (автоматически кэшируется)
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )
        print(f"   Model path: {model_path}")

        # 2. Загружаем (CPU mode)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=0,  # 0 = CPU only
            verbose=False
        )

    def generate_advice(self, user_vacancy: str, references: list) -> str:
        # Формируем контекст
        ref_text = ""
        for i, r in enumerate(references):
            ref_text += f"\n--- ПРИМЕР {i + 1} ---\n{r['title']}\n"

        system_prompt = (
            "Ты эксперт по найму. Твоя задача — дать краткие советы по улучшению вакансии, "
            "сравнивая её с успешными примерами."
        )

        user_message = (
            f"МОЯ ВАКАНСИЯ: {user_vacancy}\n\n"
            f"УСПЕШНЫЕ ПРИМЕРЫ:{ref_text}\n\n"
            "Напиши 3 конкретных совета, как улучшить мою вакансию, чтобы она была похожа на успешные. "
            "Отвечай на русском языке."
        )

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return output['choices'][0]['message']['content']


# --- TEST ---
if __name__ == "__main__":
    bot = LocalLLM()
    print("\n💬 Thinking...")
    res = bot.generate_advice("Ищем питониста", [{"title": "Senior Python (Remote)"}])
    print("\n💡 RESULT:\n", res)
