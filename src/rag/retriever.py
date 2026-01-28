import pandas as pd
import pathlib
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Optional


class VacancyRetriever:
    def __init__(self, data_path: str = None, model_name: str = "cointegrated/rubert-tiny2",
                 force_reindex: bool = False):

        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.root / "data"
        self.index_path = self.data_dir / "vector_index.pkl"

        # Создаем папку data, если её нет (на всякий случай)
        self.data_dir.mkdir(exist_ok=True)

        print(f"🔧 Initializing Retriever with model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.vacancies = []

        if not force_reindex and self.index_path.exists():
            print(f"✅ Loading vector index from {self.index_path}...")
            try:
                with open(self.index_path, "rb") as f:
                    saved_data = pickle.load(f)
                    self.index = saved_data["index"]
                    self.vacancies = saved_data["vacancies"]
            except Exception as e:
                print(f"❌ Error loading index: {e}. Rebuilding...")
                if data_path: self._build_index(data_path)
        elif data_path:
            self._build_index(data_path)
        else:
            print("⚠️ No index found and no data provided. Retrieval will be disabled.")

    def _build_index(self, data_path: str):
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return

        print(f"⚙️ Building new vector index from {data_path}...")
        try:
            df = pd.read_parquet(data_path)

            # Проверка наличия нужных колонок
            required_cols = ['is_top_performer', 'vacancy_title', 'specialization', 'skills_str', 'text_clean']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing columns in parquet. Required: {required_cols}")
                return

            # Берем только успешные
            top_df = df[df['is_top_performer'] == True].copy().reset_index(drop=True)

            if top_df.empty:
                print("⚠️ No top performers found. Using top 100 by default.")
                top_df = df.head(100).copy()

            # Создаем расширенный текст для поиска
            top_df['embedding_text'] = (
                    top_df['vacancy_title'].fillna('') + " " +
                    top_df['specialization'].fillna('') + " " +
                    top_df['skills_str'].fillna('') + " " +
                    top_df['text_clean'].fillna('').str.slice(0, 500)
            )

            vectors = self.model.encode(top_df['embedding_text'].tolist(), show_progress_bar=True)

            index = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
            index.fit(vectors)

            self.index = index
            self.vacancies = top_df.to_dict("records")

            with open(self.index_path, "wb") as f:
                pickle.dump({"index": self.index, "vacancies": self.vacancies}, f)
            print(f"✅ Index built and saved to {self.index_path}!")

        except Exception as e:
            print(f"❌ Failed to build index: {e}")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.index or not query: return []
        try:
            query_vector = self.model.encode([query])
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=limit)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.vacancies):
                    vac = self.vacancies[idx]
                    score = 1 - distances[0][i]
                    results.append({
                        "title": vac.get('vacancy_title', ''),
                        "html_text": vac.get('vacancy_description', ''),
                        "velocity": vac.get('velocity', 0),
                        "score": float(score)
                    })
            return sorted(results, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []

    def get_top_vacancy(self) -> Optional[Dict]:
        if self.vacancies:
            vac = self.vacancies[0]
            return {
                "title": vac.get('vacancy_title', ''),
                "html_text": vac.get('vacancy_description', ''),
                "velocity": vac.get('velocity', 0),
                "score": 1.0
            }
        return None
