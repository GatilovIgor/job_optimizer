import pandas as pd
import pathlib
import pickle
import logging
import warnings
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict

# --- 🔇 ТИШИНА В ЭФИРЕ ---
# Отключаем технические предупреждения HuggingFace и лишний шум
hf_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------

class VacancyRetriever:
    def __init__(self, data_path: str = None):
        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.index_path = self.root / "data" / "vector_index.pkl"

        # Модель инициализируется уже в "тихом" режиме
        self.model = SentenceTransformer("cointegrated/rubert-tiny2")
        self.index = None
        self.vacancies = []

        if self.index_path.exists():
            print("📖 Загрузка поискового индекса...")
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]
                self.vacancies = data["vacancies"]
        elif data_path:
            self._build_index(data_path)
        else:
            print("⚠️ Нет данных для поиска. RAG выключен.")

    def _build_index(self, data_path: str):
        print("⚙️ Создание индекса (векторизация)...")
        df = pd.read_parquet(data_path)

        # Фильтруем топ-перформеров
        if 'is_top_performer' in df.columns:
            top_df = df[df['is_top_performer'] == True].copy()
        else:
            top_df = df.copy()

        # Формируем текст
        top_df['embed_text'] = (
                top_df['vacancy_title'].fillna('') + " " +
                top_df['specialization'].fillna('') + " " +
                top_df['vacancy_description'].fillna('').astype(str).str.slice(0, 500)
        )

        # encode может показывать прогресс-бар, его оставим, чтобы видеть, что процесс идет
        vectors = self.model.encode(top_df['embed_text'].tolist(), show_progress_bar=True)

        self.index = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.index.fit(vectors)

        self.vacancies = top_df.to_dict("records")

        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self.index, "vacancies": self.vacancies}, f)
        print("✅ Индекс готов и сохранен.")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.index: return []

        vec = self.model.encode([query])
        distances, indices = self.index.kneighbors(vec, n_neighbors=limit)

        results = []
        for idx in indices[0]:
            if idx < len(self.vacancies):
                results.append(self.vacancies[idx])
        return results
