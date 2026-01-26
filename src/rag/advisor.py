from typing import List, Dict
import time
from src.rag.llm import LocalLLM
from src.api.models import VacancyIn, VacancyOut
from src.common.text import basic_issues, normalize_text


class VacancyAdvisor:
    def __init__(self):
        self.llm = LocalLLM()

    def _calculate_score(self, text: str, issues: list) -> int:
        """Вспомогательная функция оценки"""
        base = 100
        # Штрафы
        base -= len(issues) * 15
        # Бонусы за структуру
        if "<ul>" in text or "<li>" in text: base += 5
        if len(text) > 500: base += 5
        # Ограничения
        return max(min(base, 100), 10)

    def process_single_vacancy(self, vac_input: VacancyIn, retriever) -> VacancyOut:
        start_time = time.time()

        # 1. Анализ ИСХОДНОГО текста
        clean_input = normalize_text(vac_input.text)
        current_issues = basic_issues(clean_input)

        # Считаем оценку ДО (было)
        original_score = self._calculate_score(clean_input, current_issues)

        # 2. Поиск и Рерайт
        query = f"{vac_input.title}. {vac_input.specialization or ''}. {clean_input[:200]}"
        references = retriever.search(query, limit=1) if retriever else []

        llm_result = self.llm.generate_rewrite(
            user_vacancy={"title": vac_input.title, "text": vac_input.text},
            references=references,
            issues=current_issues
        )

        final_text = llm_result.get("rewritten_text", vac_input.text)

        # 3. Анализ НОВОГО текста
        new_issues = basic_issues(normalize_text(final_text))

        # Считаем оценку ПОСЛЕ (стало)
        final_score = self._calculate_score(final_text, new_issues)

        # Если нейросеть сделала текст длиннее и структурнее, накидываем бонусы
        if len(final_text) > len(clean_input) * 1.3: final_score += 10
        final_score = min(final_score, 100)  # Не больше 100

        return VacancyOut(
            input_id=vac_input.input_id,
            rewritten_text=final_text,
            rewrite_notes=llm_result.get("rewrite_notes", []),
            issues=current_issues,
            quality_score=int(final_score),
            original_score=int(original_score),  # Передаем старую оценку
            safety_flags=llm_result.get("safety_flags", []),
            low_confidence_retrieval=(len(references) == 0),
            debug={
                "processing_time": round(time.time() - start_time, 2),
                "top_reference": references[0]['title'] if references else None
            }
        )
