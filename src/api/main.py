import uvicorn
import pathlib
import sys
from fastapi import FastAPI
from contextlib import asynccontextmanager

# При запуске через -m src.api.main Python сам добавит корень в path,
# но на всякий случай явно укажем корень проекта
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.api.models import RewriteRequest, RewriteResponse
from src.rag.retriever import VacancyRetriever
from src.rag.llm import VacancyOptimizer

retriever = None
optimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, optimizer
    data_path = root_dir / "data" / "vacancies_processed.parquet"

    print("🚀 Инициализация AI ядра...")
    retriever = VacancyRetriever(str(data_path) if data_path.exists() else None)
    optimizer = VacancyOptimizer()
    yield
    print("🛑 Остановка ядра.")


app = FastAPI(lifespan=lifespan)


@app.post("/optimize", response_model=RewriteResponse)
async def optimize_endpoint(req: RewriteRequest):
    results = []
    for vac in req.vacancies:
        # Поиск референсов
        query = f"{vac.vacancy_title} {vac.specialization}"
        refs = retriever.search(query) if retriever else []

        # Генерация
        res = optimizer.optimize(vac, refs)
        results.append(res)
    return RewriteResponse(results=results)


if __name__ == "__main__":
    # Настройки для локального запуска
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
