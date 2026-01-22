from fastapi import FastAPI
from src.api.models import RewriteRequest, RewriteResponse, VacancyOut
from src.common.text import normalize_text, basic_issues, heuristic_quality_score

app = FastAPI(title="AI Job Optimizer MVP", version="0.1.0")

@app.post("/rewrite-batch", response_model=RewriteResponse)
def rewrite_batch(req: RewriteRequest) -> RewriteResponse:
    results = []
    for v in req.vacancies:
        text = normalize_text(v.text)
        issues = basic_issues(text)
        score = heuristic_quality_score(text, issues)

        # пока: заглушка rewrite (в следующем шаге заменим на retrieval+LLM)
        rewritten = text

        results.append(VacancyOut(
            input_id=v.input_id,
            rewritten_text=rewritten,
            rewrite_notes=["MVP: пока без LLM. Добавим после retrieval."],
            issues=issues,
            quality_score=score,
            safety_flags=[],
            low_confidence_retrieval=True,
            debug={"objective": req.objective, "tone": req.tone}
        ))
    return RewriteResponse(results=results)
