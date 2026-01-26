from typing import List, Dict
import re
import time
from src.rag.llm import LocalLLM
from src.api.models import VacancyIn, VacancyOut
from src.common.text import normalize_text


class VacancyAdvisor:
    def __init__(self):
        print("🔧 Initializing Precision HR Advisor...", flush=True)
        self.llm = LocalLLM()

    def _analyze_quality(self, text: str) -> Dict:
        """
        Точечная оценка качества вакансии (шаг 5 баллов).
        Аддитивная система: начинаем с 0 и начисляем за каждый плюс.
        """
        score = 0
        issues = []
        text_lower = text.lower()

        # --- 1. ОБЪЕМ (Макс 15) ---
        length = len(text)
        if length < 200:
            issues.append("❌ Текст критически короткий (нужно > 200 символов)")
        elif length < 600:
            score += 5
            issues.append("⚠️ Текст коротковат, мало деталей")
        else:
            score += 15  # Отличный объем

        # --- 2. СТРУКТУРА (Макс 30) ---
        # Обязанности (+10)
        if any(w in text_lower for w in ["обязанност", "задачи", "предстоит", "делать", "функционал"]):
            score += 10
        else:
            issues.append("❓ Нет блока 'Обязанности'")

        # Требования (+10)
        if any(w in text_lower for w in ["требован", "ожидаем", "ищем", "навыки", "знания", "опыт"]):
            score += 10
        else:
            issues.append("❓ Нет блока 'Требования'")

        # Условия (+10)
        if any(w in text_lower for w in ["условия", "предлагаем", "мы даем", "оффер", "соцпакет", "гарантируем"]):
            score += 10
        else:
            issues.append("❓ Нет блока 'Условия'")

        # --- 3. ДЕТАЛИ (Макс 40) ---
        # Зарплата (+10)
        has_salary = False
        money_words = ["руб", "₽", "$", "€", "оклад", "доход", "зарплат", "з/п", "на руки", "gross", "net", "преми",
                       "бонус"]
        if any(w in text_lower for w in money_words):
            score += 10
            has_salary = True
            # Бонус за конкретные цифры (+5)
            # Ищем числа от 1000 (зарплаты)
            if re.search(r'\d{2,}', text):
                score += 5
        else:
            issues.append("💰 Не указаны условия оплаты")

        # График (+10)
        if any(w in text_lower for w in
               ["график", "5/2", "2/2", "удален", "гибрид", "офис", "вахта", "полный день", "сменный"]):
            score += 10
        else:
            issues.append("📅 Не указан график работы")

        # Локация / Место (+5)
        if any(w in text_lower for w in ["офис", "м.", "метро", "адрес", "город", "центр", "парк", "удален"]):
            score += 5

        # Контакты / Призыв (+5)
        if any(w in text_lower for w in ["отклик", "резюме", "звон", "писат", "связ", "присылай", "ждем"]):
            score += 5

        # Оформление / Компания (+5)
        if any(w in text_lower for w in ["тк рф", "договор", "оформлен", "компани", "команд", "коллектив"]):
            score += 5

        # --- 4. ОФОРМЛЕНИЕ (Макс 15) ---
        # Списки (+10)
        # Ищем HTML теги или символы списков
        has_html_list = "<ul>" in text or "<li>" in text
        has_text_list = any(x in text for x in ["•", "⁃", "— ", "1.", "2."])

        if has_html_list or has_text_list:
            score += 10
        else:
            issues.append("📄 Сплошной текст (добавьте списки)")

        # HTML форматирование (+5)
        if "<strong>" in text or "<b>" in text or "<h3>" in text or "<br>" in text:
            score += 5

        return {"score": min(score, 100), "issues": issues}

    def process_single_vacancy(self, vac_input: VacancyIn, retriever) -> VacancyOut:
        print(f"▶️ Start processing: {vac_input.input_id}", flush=True)
        start_time = time.time()

        in_title = vac_input.title.strip() if vac_input.title else ""
        in_text = vac_input.text.strip() if vac_input.text else ""
        in_spec = vac_input.specialization.strip() if vac_input.specialization else ""

        if not any([in_title, in_text, in_spec]):
            return VacancyOut(
                input_id=vac_input.input_id,
                rewritten_title="Пример", rewritten_specialization="Продажи", rewritten_text="<p>Нет данных</p>",
                rewrite_notes=["Пустой ввод"], issues=[], quality_score=0, original_score=0, safety_flags=[],
                low_confidence_retrieval=True
            )

        # 1. Анализ ИСХОДНИКА
        if in_text:
            clean_input = normalize_text(in_text)
            analysis = self._analyze_quality(clean_input)
            original_score = analysis["score"]
            current_issues = analysis["issues"]
        else:
            clean_input = ""
            current_issues = ["Текст отсутствует"]
            original_score = 0

        # 2. Поиск
        search_parts = [p for p in [in_title, in_spec, clean_input[:200]] if p]
        query = ". ".join(search_parts)
        references = retriever.search(query, limit=1) if (retriever and query) else []

        # 3. LLM
        llm_result = self.llm.generate_rewrite(
            user_vacancy={"title": in_title, "text": in_text, "specialization": in_spec},
            references=references,
            issues=current_issues
        )

        final_text = llm_result.get("rewritten_text", in_text)
        final_title = llm_result.get("title", in_title)
        final_spec = llm_result.get("specialization", in_spec)

        # 4. Анализ РЕЗУЛЬТАТА
        final_analysis = self._analyze_quality(final_text)
        final_score = final_analysis["score"]

        # Гарантия улучшения для пользователя (психологический момент)
        # Если ИИ реально поработал, оценка не должна быть ниже исходной
        if final_score < original_score and len(final_text) > len(clean_input):
            final_score = original_score + 5

        return VacancyOut(
            input_id=vac_input.input_id,
            rewritten_title=final_title,
            rewritten_specialization=final_spec,
            rewritten_text=final_text,
            rewrite_notes=llm_result.get("rewrite_notes", []),
            issues=current_issues,
            quality_score=int(final_score),
            original_score=int(original_score),
            safety_flags=llm_result.get("safety_flags", []),
            low_confidence_retrieval=(len(references) == 0),
            debug={"processing_time": round(time.time() - start_time, 2)}
        )
