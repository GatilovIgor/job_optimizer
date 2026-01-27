from typing import List, Dict
import re
import html
import time
from src.rag.llm import LocalLLM
from src.api.models import VacancyIn, VacancyOut
from src.common.text import normalize_text


class VacancyAdvisor:
    def __init__(self):
        print("🔧 Initializing Single-Field Advisor...", flush=True)
        self.llm = LocalLLM()

    def _clean_html(self, raw_text: str) -> str:
        """Очищает текст от HTML-тегов и спецсимволов"""
        if not raw_text: return ""
        text = html.unescape(raw_text)
        # Заменяем структурные теги на переносы
        text = re.sub(r'<li>', '\n• ', text)
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'</p>|</div>', '\n\n', text)
        # Удаляем все остальные теги
        text = re.sub(r'<[^>]+>', '', text)
        # Убираем лишние пробелы
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def _analyze_quality(self, text: str) -> Dict:
        """Анализ качества (0-100)"""
        score = 0
        issues = []
        text_lower = text.lower()

        # Если текста нет совсем или он очень короткий (генерируем с нуля)
        if len(text) < 50:
            return {"score": 0, "issues": ["Текст отсутствует (будет сгенерирован)"]}

        # 1. ОБЪЕМ
        if len(text) < 200:
            issues.append("❌ Критически мало текста")
        elif len(text) < 600:
            score += 5; issues.append("⚠️ Мало деталей")
        else:
            score += 15

        # 2. СТРУКТУРА
        for kw in ["обязанност", "задачи", "делат"]:
            if kw in text_lower: score += 10; break
        else:
            issues.append("❓ Нет блока 'Обязанности'")

        for kw in ["требован", "ищем", "навыки"]:
            if kw in text_lower: score += 10; break
        else:
            issues.append("❓ Нет блока 'Требования'")

        for kw in ["условия", "предлагаем", "оффер"]:
            if kw in text_lower: score += 10; break
        else:
            issues.append("❓ Нет блока 'Условия'")

        # 3. ДЕТАЛИ
        money_words = ["руб", "₽", "оклад", "доход", "зарплат", "на руки", "gross", "net"]
        if any(w in text_lower for w in money_words):
            score += 10
            if re.search(r'\d{2,}', text): score += 5  # Цифры
        else:
            issues.append("💰 Не указана зарплата")

        if any(w in text_lower for w in ["график", "5/2", "2/2", "удален", "офис", "сменный"]):
            score += 10
        else:
            issues.append("📅 Не указан график")

        if any(w in text_lower for w in ["офис", "м.", "город", "адрес"]): score += 5
        if any(w in text_lower for w in ["связ", "звон", "писат", "отклик"]): score += 5
        if any(w in text_lower for w in ["тк рф", "оформлен"]): score += 5

        # 4. ОФОРМЛЕНИЕ
        if "<ul>" in text or "•" in text or "— " in text:
            score += 10
        else:
            issues.append("📄 Нет списков")
        if "<b>" in text or "<strong>" in text: score += 5

        return {"score": min(score, 100), "issues": issues}

    def process_single_vacancy(self, vac_input: VacancyIn, retriever) -> VacancyOut:
        print(f"▶️ Start processing: {vac_input.input_id}", flush=True)
        start_time = time.time()

        # 1. Очистка и Нормализация
        in_title = vac_input.title.strip() if vac_input.title else ""
        in_spec = vac_input.specialization.strip() if vac_input.specialization else ""

        # Чистим HTML если он есть в input
        raw_text = vac_input.text if vac_input.text else ""
        in_text = self._clean_html(raw_text)

        # Если вообще всё пусто
        if not any([in_title, in_text, in_spec]):
            return VacancyOut(
                input_id=vac_input.input_id,
                rewritten_title="Пример", rewritten_specialization="IT", rewritten_text="<p>Пустой запрос</p>",
                rewrite_notes=["Введите хотя бы что-то"], issues=[], quality_score=0, original_score=0, safety_flags=[],
                low_confidence_retrieval=True
            )

        # 2. Анализ ИСХОДНИКА
        analysis = self._analyze_quality(in_text)
        original_score = analysis["score"]
        current_issues = analysis["issues"]

        # 3. Поиск референсов (RAG)
        # Ищем по тому, что есть
        search_query = f"{in_title} {in_spec} {in_text[:200]}"
        references = retriever.search(search_query, limit=1) if (retriever and search_query.strip()) else []

        # 4. LLM Генерация
        # Нейросеть сама поймет, что заполнить
        llm_result = self.llm.generate_rewrite(
            user_vacancy={"title": in_title, "text": in_text, "specialization": in_spec},
            references=references,
            issues=current_issues
        )

        final_text = llm_result.get("rewritten_text", in_text)
        final_title = llm_result.get("title", in_title)
        final_spec = llm_result.get("specialization", in_spec)

        # 5. Анализ РЕЗУЛЬТАТА
        final_analysis = self._analyze_quality(final_text)
        final_score = final_analysis["score"]

        # Если генерировали с нуля (было пусто, стало много) -> ставим высокую оценку
        if len(in_text) < 50 and len(final_text) > 500:
            final_score = max(final_score, 90)

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
