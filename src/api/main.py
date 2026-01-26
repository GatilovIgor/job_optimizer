import sys
import pathlib
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

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
    data_path = root_dir / "dataset" / "vacancies_processed.parquet"

    try:
        retriever = VacancyRetriever(data_path=str(data_path) if data_path.exists() else None)
        advisor = VacancyAdvisor()
        print("✅ AI Ready!")
    except Exception as e:
        print(f"❌ Init Error: {e}")
    yield
    print("🛑 AI Stopped.")


app = FastAPI(lifespan=lifespan)


@app.post("/rewrite-batch", response_model=RewriteResponse)
async def rewrite_batch(req: RewriteRequest):
    if not advisor:
        raise HTTPException(status_code=503, detail="AI loading...")

    results = []
    print(f"📥 Processing {len(req.vacancies)} vacancies...")

    for vac in req.vacancies:
        try:
            # Основная обработка
            res = advisor.process_single_vacancy(vac, retriever)
            results.append(res)
        except Exception as e:
            print(f"❌ Error processing vacancy {vac.input_id}: {e}")
            # Возвращаем заглушку с ошибкой
            error_res = VacancyOut(
                input_id=vac.input_id,
                rewritten_text=f"Ошибка обработки: {str(e)}",
                rewrite_notes=["Internal Error"],
                issues=[],
                quality_score=0,
                safety_flags=[],
                low_confidence_retrieval=True
            )
            results.append(error_res)

    return RewriteResponse(results=results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
