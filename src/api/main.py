import sys
import pathlib
import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Настройка путей для импорта модулей
# Берем путь к файлу, поднимаемся на 3 уровня вверх до корня
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.rag.retriever import VacancyRetriever
from src.rag.advisor import VacancyAdvisor
from src.api.models import RewriteRequest, RewriteResponse, VacancyOut

retriever = None
advisor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, advisor
    print("🚀 Starting AI Engine...")

    # Правильный путь к данным
    data_path = root_dir / "data" / "vacancies_processed.parquet"

    if not data_path.exists():
        print(f"⚠️ Warning: Dataset not found at {data_path}")
        print("   Run 'python src/data/prepare.py' first!")

    try:
        # Инициализируем компоненты
        # Передаем путь только если файл существует
        retriever = VacancyRetriever(data_path=str(data_path) if data_path.exists() else None)
        advisor = VacancyAdvisor()
        print("✅ AI Ready!")
    except Exception as e:
        print(f"❌ Init Error: {e}")
        # Не роняем приложение, чтобы оно могло запуститься хотя бы для healthcheck
    yield
    print("🛑 AI Stopped.")


app = FastAPI(lifespan=lifespan, title="Job Optimizer API")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Job Optimizer API is running"}


@app.post("/rewrite-batch", response_model=RewriteResponse)
async def rewrite_batch(req: RewriteRequest):
    if not advisor:
        raise HTTPException(status_code=503, detail="AI Engine is loading or failed to init.")

    results = []
    print(f"📥 Processing batch of {len(req.vacancies)} vacancies...")

    for vac in req.vacancies:
        try:
            # Основная логика
            res = advisor.process_single_vacancy(vac, retriever)
            results.append(res)
        except Exception as e:
            print(f"❌ Error processing vacancy {vac.input_id}: {e}")
            # Возвращаем безопасную заглушку при ошибке
            error_res = VacancyOut(
                input_id=vac.input_id,
                rewritten_title=vac.title or "Error",
                rewritten_specialization=vac.specialization or "Unknown",
                rewritten_text=f"Ошибка обработки: {str(e)}",
                rewrite_notes=["Internal Server Error"],
                issues=[],
                quality_score=0,
                original_score=0,
                safety_flags=[],
                low_confidence_retrieval=True
            )
            results.append(error_res)

    return RewriteResponse(results=results)


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)
