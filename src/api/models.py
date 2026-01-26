from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- Входные модели (Input) ---

class VacancyIn(BaseModel):
    input_id: str = "default"
    title: str
    text: str

class RewriteRequest(BaseModel):
    # API ожидает список вакансий (batch mode)
    vacancies: List[VacancyIn]

class AnalyzeRequest(BaseModel):
    # Старый формат запроса (для совместимости)
    title: str
    description: Optional[str] = None

# --- Выходные модели (Output) ---

class VacancyOut(BaseModel):
    input_id: str
    rewritten_text: str
    rewrite_notes: List[str]
    issues: List[str]
    quality_score: int
    safety_flags: List[str]
    low_confidence_retrieval: bool
    debug: Optional[Dict[str, Any]] = None

class RewriteResponse(BaseModel):
    # Ответ содержит список обработанных результатов
    results: List[VacancyOut]
