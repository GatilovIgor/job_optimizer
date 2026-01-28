from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VacancyIn(BaseModel):
    input_id: str = "default"
    profile: Optional[str] = Field(default="", description="Профиль (например, Менеджмент)")
    city: Optional[str] = Field(default="", description="Город")
    vacancy_title: str = Field(..., description="Название вакансии")
    vacancy_description: str = Field(..., description="Текст вакансии")
    specialization: Optional[str] = Field(default="", description="Специализация")


class VacancyOut(BaseModel):
    input_id: str
    # 5 улучшенных полей
    profile: str
    city: str
    vacancy_title: str
    vacancy_description: str
    specialization: str

    # Метаданные
    efficiency_prediction: Optional[float] = Field(description="Прогнозируемая эффективность")
    improvement_notes: List[str] = Field(default_factory=list)


class RewriteRequest(BaseModel):
    vacancies: List[VacancyIn]


class RewriteResponse(BaseModel):
    results: List[VacancyOut]
