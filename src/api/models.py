from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class VacancyIn(BaseModel):
    input_id: str = "default"
    title: str
    text: str
    specialization: Optional[str] = None # Новое: для точности поиска
    skills: Optional[List[str]] = None   # Новое: для точности поиска

class RewriteRequest(BaseModel):
    vacancies: List[VacancyIn]

class AnalyzeRequest(BaseModel):
    title: str
    description: Optional[str] = None

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
    results: List[VacancyOut]
