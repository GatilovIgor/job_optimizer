from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class VacancyIn(BaseModel):
    input_id: str = Field(..., description="ID строки/вакансии во входном батче")
    title: Optional[str] = None
    text: str = Field(..., description="Основной текст вакансии (description/body)")
    language_hint: Optional[str] = Field(None, description="ru/en если известно")


class RewriteRequest(BaseModel):
    vacancies: List[VacancyIn]
    objective: str = Field("mixed", description="applications|views|conversion|mixed")
    tone: str = Field("neutral", description="neutral|friendly|formal|short")
    max_refs: int = Field(3, ge=1, le=5)


class VacancyOut(BaseModel):
    input_id: str
    rewritten_text: str
    rewrite_notes: List[str]
    issues: List[str]
    quality_score: int  # 0..100
    safety_flags: List[str]
    low_confidence_retrieval: bool
    debug: Dict[str, Any] = Field(default_factory=dict)


class RewriteResponse(BaseModel):
    results: List[VacancyOut]
