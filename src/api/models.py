from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union


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
    rewritten_title: str = Field(description="Сгенерированный или улучшенный заголовок")
    rewritten_specialization: str = Field(description="Определенная сфера")
    rewritten_text: str

    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    rewrite_notes: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    # -------------------------

    quality_score: int
    original_score: int
    low_confidence_retrieval: bool
    debug: Optional[Dict[str, Any]] = None

    # Добавляем валидатор, который превращает строку в список, если нужно
    @field_validator('rewrite_notes', 'issues', 'safety_flags', mode='before')
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]  # Превращаем "Ошибка" в ["Ошибка"]
        return v


class RewriteResponse(BaseModel):
    results: List[VacancyOut]
