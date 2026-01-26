import sys
import pathlib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- МАГИЯ ПУТЕЙ (ВСТАВИТЬ В САМОЕ НАЧАЛО) ---
# Находим папку "job-optimizer-mvp" (на 2 уровня выше файла main.py)
root_dir = pathlib.Path(__file__).resolve().parent.parent.parent
# Добавляем её в путь поиска модулей
sys.path.append(str(root_dir))
# ---------------------------------------------

# Теперь импорты точно заработают
from src.rag.retriever import VacancyRetriever
from src.rag.advisor import VacancyAdvisor

app = FastAPI(title="Job Optimizer AI")

retriever = None
advisor = None


@app.on_event("startup")
async def startup_event():
    global retriever, advisor
    print("🚀 Initializing AI Engine...")

    data_path = root_dir / "dataset" / "vacancies_processed.parquet"
    if not data_path.exists():
        print(f"⚠️ Warning: Dataset not found at {data_path}")

    retriever = VacancyRetriever(data_path=str(data_path) if data_path.exists() else None)
    advisor = VacancyAdvisor()
    print("✅ AI Ready!")


class AnalyzeRequest(BaseModel):
    title: str
    description: str = ""


@app.post("/analyze")
async def analyze_vacancy(req: AnalyzeRequest):
    if not retriever or not advisor:
        raise HTTPException(status_code=503, detail="Loading models...")

    print(f"🔍 Analyzing: {req.title}")

    # 1. RAG
    query = f"{req.title} {req.description}"
    champions = retriever.search(query, limit=3)

    # 2. LLM
    analysis = advisor.analyze(query, champions)

    return {
        "input": req.title,
        "advice": analysis.get('ai_advice_text', 'No advice generated'),
        "similar_top_cases": [c['title'] for c in champions]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
