import pandas as pd
import pathlib
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class VacancyRetriever:
    def __init__(self, data_path: str = None):
        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.root / "data"
        self.index_path = self.data_dir / "vector_index_optimized.pkl"

        # Модель
        self.model = SentenceTransformer("cointegrated/rubert-tiny2")
        self.index = None
        self.vacancies = []

        if self.index_path.exists():
            self._load_index()
        elif data_path:
            self._build_index(data_path)
        else:
            print("⚠️ Retriever не инициализирован данными.")

    def _load_index(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.vacancies = data["vacancies"]
        print("✅ Индекс загружен.")

    def _build_index(self, data_path):
        print("⚙️ Строим индекс по Top Performers...")
        df = pd.read_parquet(data_path)

        # Берем только лучших
        top_df = df[df['is_top_performer'] == True].copy()

        # Формируем текст для эмбеддинга: Заголовок + Спец + Описание
        top_df['embed_text'] = (
                top_df['vacancy_title'].fillna('') + " " +
                top_df['specialization'].fillna('') + " " +
                top_df['vacancy_description'].fillna('').str.slice(0, 1000)
        )

        vectors = self.model.encode(top_df['embed_text'].tolist(), show_progress_bar=True)

        self.index = NearestNeighbors(n_neighbors=3, metric="cosine")
        self.index.fit(vectors)
        self.vacancies = top_df.to_dict("records")

        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self.index, "vacancies": self.vacancies}, f)
        print("✅ Индекс построен.")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.index: return []
        vec = self.model.encode([query])
        distances, indices = self.index.kneighbors(vec, n_neighbors=limit)

        results = []
        for idx in indices[0]:
            if idx < len(self.vacancies):
                results.append(self.vacancies[idx])
        return results
