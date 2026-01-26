import sys
import pathlib
import uvicorn
from fastapi import FastAPI, HTTPException

# --- МАГИЯ ПУТЕЙ ---
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))
# -------------------

from src.rag.retriever import VacancyRetriever
from src.rag.advisor import VacancyAdvisor
from src.api.models import RewriteRequest, RewriteResponse, VacancyIn, AnalyzeRequest

app = FastAPI(title="Job Optimizer AI (MVP)")

retriever = None
advisor = None


@app.on_event("startup")
async def startup_event():
    global retriever, advisor
    print("🚀 Initializing AI Engine...")

    # Пытаемся найти датасет
    data_path = root_dir / "dataset" / "vacancies_processed.parquet"
    if not data_path.exists():
        print(f"⚠️ Warning: Dataset not found at {data_path}. RAG will be empty.")
        retriever_path = None
    else:
        retriever_path = str(data_path)

    # Инициализация (загрузит модель в память)
    retriever = VacancyRetriever(data_path=retriever_path)
    advisor = VacancyAdvisor()
    print("✅ AI Ready and Loaded!")


@app.post("/rewrite-batch", response_model=RewriteResponse)
async def rewrite_batch_vacancies(req: RewriteRequest):
    """
    Основной метод MVP: принимает список вакансий, возвращает улучшенные версии.
    """
    if not advisor:
        raise HTTPException(status_code=503, detail="AI System is still loading...")

    results = []
    print(f"📥 Batch request received: {len(req.vacancies)} items.")

    for vac_in in req.vacancies:
        # Обработка каждой вакансии
        # В MVP делаем это последовательно.
        # При масштабировании здесь будет очередь задач (Celery/Redis).
        try:
            result = advisor.process_single_vacancy(vac_in, retriever)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {vac_in.input_id}: {e}")
            # Возвращаем ошибку в структуре, чтобы не валить весь батч
            # (упрощенная обработка ошибок)
            continue

    return RewriteResponse(results=results)


# --- Legacy endpoint (можно оставить для совместимости с app.py, если нужно) ---
@app.post("/analyze")
async def analyze_legacy(req: AnalyzeRequest):
    # Преобразуем старый запрос в новый формат
    vac_in = VacancyIn(
        input_id="legacy_1",
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
