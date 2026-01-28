import os
import pathlib

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

class LocalLLM:
    def __init__(self, n_ctx=4096):
        if Llama is None:
            raise ImportError("Please install llama-cpp-python")

        # 1. Находим путь к модели
        # Поднимаемся от src/rag/llm.py до корня проекта
        root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
        model_path = root_dir / "models" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

        print(f"📦 Loading local model: {model_path}...", flush=True)

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model file not found at: {model_path}\nPlease put the .gguf file in the 'models' folder.")

        try:
            self.llm = Llama(
                model_path=str(model_path), # Путь к вашему файлу
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                n_threads=6,
                verbose=False
            )
            print("📦 Llama-3.2-3B Loaded!", flush=True)
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            self.llm = None

    def generate_rewrite(self, user_vacancy: dict, references: list, issues: list) -> dict:
        if not self.llm:
            return {"raw_response": "Ошибка: Модель не загружена"}

        print("      [Llama] Generating...", flush=True)

        title = user_vacancy.get('title', 'Сотрудник')
        text = user_vacancy.get('text', '')

        if len(text) < 100:
            text += f"\n(Информация скупая. Придумай профессиональные обязанности и требования для роли '{title}')"

        system_prompt = (
            "You are a professional HR Specialist. Output MUST be in Russian.\n"
            "Follow the structure exactly."
        )

        user_message = (
            f"Напиши подробную вакансию: {title}.\n"
            f"Черновик: {text}\n\n"
            "СТРУКТУРА ОТВЕТА:\n"
            "ЗАГОЛОВОК: [Должность]\n"
            "СФЕРА: [Сфера]\n"
            "ОПИСАНИЕ:\n"
            "<p>[Вступление]</p>\n"
            "<h3>Обязанности:</h3>\n"
            "<ul>\n"
            "<li>[Пункт 1]</li>\n"
            "<li>[Пункт 2]</li>\n"
            "<li>[Пункт 3]</li>\n"
            "<li>[Пункт 4]</li>\n"
            "<li>[Пункт 5]</li>\n"
            "</ul>\n"
            "<h3>Требования:</h3>\n"
            "<ul>\n"
            "<li>[Пункт 1]</li>\n"
            "<li>[Пункт 2]</li>\n"
            "<li>[Пункт 3]</li>\n"
            "</ul>\n"
            "<h3>Условия:</h3>\n"
            "<ul>\n"
            "<li>[Зарплата]</li>\n"
            "<li>[График]</li>\n"
            "<li>[Офис/Бонусы]</li>\n"
            "</ul>"
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return {"raw_response": response['choices'][0]['message']['content']}
        except Exception as e:
            print(f"      ❌ LLM Gen Error: {e}", flush=True)
            return {"raw_response": text}
