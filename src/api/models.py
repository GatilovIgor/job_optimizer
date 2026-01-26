from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class VacancyIn(BaseModel):
    input_id: str = "default"
    title: Optional[str] = Field(default="", description="Название должности")
    text: Optional[str] = Field(default="", description="Текст вакансии")
    specialization: Optional[str] = Field(default="", description="Специализация")
    skills: Optional[List[str]] = Field(default_factory=list)

class RewriteRequest(BaseModel):
    vacancies: List[VacancyIn]

class VacancyOut(BaseModel):
    input_id: str
    # --- НОВЫЕ ПОЛЯ ---
    rewritten_title: str = Field(description="Сгенерированный или улучшенный заголовок")
    rewritten_specialization: str = Field(description="Определенная сфера")
    # ------------------
    rewritten_text: str
    rewrite_notes: List[str]
    issues: List[str]
    quality_score: int
    original_score: int
    safety_flags: List[str]
    low_confidence_retrieval: bool
    debug: Optional[Dict[str, Any]] = None

class RewriteResponse(BaseModel):
    results: List[VacancyOut]
