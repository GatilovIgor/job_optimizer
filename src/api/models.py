from pydantic import BaseModel, Field
from typing import List, Optional


class VacancyIn(BaseModel):
    input_id: str = Field(default="default", description="ID для отслеживания")
    profile: str = Field(..., description="Профиль деятельности")
    city: str = Field(..., description="Город")
    vacancy_title: str = Field(..., description="Название должности")
    vacancy_description: str = Field(..., description="Текст вакансии")
    specialization: str = Field(..., description="Специализация")


class VacancyOut(BaseModel):
    input_id: str
    profile: str
    city: str
    vacancy_title: str
    vacancy_description: str
    specialization: str

    improvement_notes: List[str] = Field(default_factory=list)
    # Используем то же имя, что в llm.py
    predicted_efficiency_score: Optional[float] = Field(default=None)


class RewriteRequest(BaseModel):
    vacancies: List[VacancyIn]


class RewriteResponse(BaseModel):
    results: List[VacancyOut]
