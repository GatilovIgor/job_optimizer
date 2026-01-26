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

        clean_text = normalize_text(vac_input.text)
        current_issues = basic_issues(clean_text)

        query = f"{vac_input.title or ''} {clean_text[:500]}"
        references = retriever.search(query, limit=3) if retriever else []

        user_vac_dict = {"title": vac_input.title, "text": clean_text}

        llm_result = self.llm.generate_rewrite(
            user_vacancy=user_vac_dict,
            references=references,
            issues=current_issues
        )

        final_text = llm_result.get("rewritten_text", clean_text)
        new_issues = basic_issues(final_text)
        quality_score = heuristic_quality_score(final_text, new_issues)

        return VacancyOut(
            input_id=vac_input.input_id,
            rewritten_text=final_text,
            rewrite_notes=llm_result.get("rewrite_notes", []),
            issues=current_issues,
            quality_score=quality_score,
            safety_flags=llm_result.get("safety_flags", []),
            low_confidence_retrieval=(len(references) == 0),
            debug={
                "processing_time": round(time.time() - start_time, 2),
                "top_reference": references[0]['title'] if references else None
            }
        )
