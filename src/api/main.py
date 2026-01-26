import sys
import pathlib
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# --- МАГИЯ ПУТЕЙ ---
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
# -------------------

from src.rag.retriever import VacancyRetriever
from src.rag.advisor import VacancyAdvisor
from src.api.models import RewriteRequest, RewriteResponse, VacancyIn, AnalyzeRequest

# Глобальные переменные
retriever = None
advisor = None


# --- НОВЫЙ СПОСОБ ЗАПУСКА (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Эта функция выполняется один раз при старте приложения.
    Здесь мы загружаем тяжелые нейросети в память.
    """
    global retriever, advisor
    print("🚀 Initializing AI Engine (Lifespan)...")

    # 1. Проверяем наличие датасета
    data_path = root_dir / "dataset" / "vacancies_processed.parquet"
    retriever_path = None

    if data_path.exists():
        retriever_path = str(data_path)
    else:
        print(f"⚠️ Warning: Dataset not found at {data_path}. RAG will be empty.")

    # 2. Инициализация моделей
    try:
        retriever = VacancyRetriever(data_path=retriever_path)
        advisor = VacancyAdvisor()
        print("✅ AI Ready and Loaded!")
    except Exception as e:
        print(f"❌ Error initializing AI: {e}")

    # Здесь приложение работает...
    yield

    # Здесь код выполнится при выключении (очистка памяти и т.д.)
    print("🛑 AI Engine stopped.")


# Подключаем lifespan в приложение
app = FastAPI(title="Job Optimizer AI (MVP)", lifespan=lifespan)


@app.post("/rewrite-batch", response_model=RewriteResponse)
async def rewrite_batch_vacancies(req: RewriteRequest):
    """
    Основной метод: принимает список вакансий -> возвращает список улучшенных.
    """
    if not advisor:
        raise HTTPException(status_code=503, detail="AI System is still loading...")

    results = []
    print(f"📥 Batch request received: {len(req.vacancies)} items.")

    for vac_in in req.vacancies:
        try:
            # Запускаем полный цикл обработки
            result = advisor.process_single_vacancy(vac_in, retriever)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {vac_in.input_id}: {e}")
            continue

    return RewriteResponse(results=results)


@app.post("/analyze")
async def analyze_legacy(req: AnalyzeRequest):
    """
    Упрощенный метод (Legacy).
    """
    if not advisor:
        raise HTTPException(status_code=503, detail="AI System is still loading...")

    vac_in = VacancyIn(
        input_id="legacy_request",
        title=req.title,
        text=req.description or req.title
    )

    res = advisor.process_single_vacancy(vac_in, retriever)

    return {
        "input": req.title,
        "advice": "\n".join(res.rewrite_notes),
        "similar_top_cases": [res.debug.get("top_reference", "N/A")]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
