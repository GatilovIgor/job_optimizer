from typing import List, Dict
import time
from src.rag.llm import LocalLLM
from src.api.models import VacancyIn, VacancyOut
from src.common.text import basic_issues, heuristic_quality_score, normalize_text


class VacancyAdvisor:
    def __init__(self):
        self.llm = LocalLLM()

    def process_single_vacancy(self, vac_input: VacancyIn, retriever) -> VacancyOut:
        start_time = time.time()

        # Анализ исходника
        clean_text = normalize_text(vac_input.text)
        current_issues = basic_issues(clean_text)

        # Поиск референса
        query = f"{vac_input.title} {clean_text[:300]}"
        references = retriever.search(query, limit=1) if retriever else []

        # Генерация
        llm_result = self.llm.generate_rewrite(
            user_vacancy={"title": vac_input.title, "text": vac_input.text},
            references=references,
            issues=current_issues
        )

        final_text = llm_result.get("rewritten_text", vac_input.text)

        # Проверяем финальный текст на наличие проблем
        # Если проблем 0 — оценка будет 100
        final_issues = basic_issues(normalize_text(final_text))

        # Расчет оценки: если проблем нет, даем 100. Если есть - вычитаем за каждую.
        quality_score = 100 - (len(final_issues) * 15)
        quality_score = max(min(quality_score, 100), 10)  # Ограничиваем от 10 до 100

        return VacancyOut(
            input_id=vac_input.input_id,
            rewritten_text=final_text,
            rewrite_notes=llm_result.get("rewrite_notes", []),
            issues=current_issues,
            quality_score=int(quality_score),
            safety_flags=llm_result.get("safety_flags", []),
            low_confidence_retrieval=(len(references) == 0),
            debug={
                "processing_time": round(time.time() - start_time, 2),
                "top_reference": references[0]['title'] if references else None
            }
        )
