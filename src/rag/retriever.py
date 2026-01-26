import pandas as pd
import pathlib
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class VacancyRetriever:
    def __init__(self, data_path: str = None, model_name: str = "cointegrated/rubert-tiny2",
                 force_reindex: bool = False):
        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.index_path = self.root / "dataset" / "vector_index.pkl"
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.vacancies = []

        if not force_reindex and self.index_path.exists():
            print(f"✅ Loading vector index from {self.index_path}...")
            with open(self.index_path, "rb") as f:
                saved_data = pickle.load(f)
                self.index = saved_data["index"]
                self.vacancies = saved_data["vacancies"]
        elif data_path:
            self._build_index(data_path)
        else:
            print("⚠️ No index found.")

    def _build_index(self, data_path: str):
        print(f"⚙️ Building new vector index from {data_path}...")
        df = pd.read_parquet(data_path)

        # Берем только успешные
        top_df = df[df['is_top_performer'] == True].copy().reset_index(drop=True)

        # Создаем расширенный текст для поиска
        top_df['embedding_text'] = (
                top_df['vacancy_title'].fillna('') + " " +
                top_df['specialization'].fillna('') + " " +
                top_df['skills_str'].fillna('') + " " +
                top_df['text_clean'].fillna('').str.slice(0, 500)  # Берем начало текста
        )

        vectors = self.model.encode(top_df['embedding_text'].tolist(), show_progress_bar=True)
        index = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
        index.fit(vectors)

        self.index = index
        self.vacancies = top_df.to_dict("records")

        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self.index, "vacancies": self.vacancies}, f)
        print("✅ Index built!")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.index or not query: return []
        query_vector = self.model.encode([query])
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=limit)

        results = []
        for i, idx in enumerate(indices[0]):
            vac = self.vacancies[idx]
            score = 1 - distances[0][i]
            results.append({
                "title": vac['vacancy_title'],
                "html_text": vac['vacancy_description'],
                "velocity": vac.get('velocity', 0),
                "score": float(score)
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)
