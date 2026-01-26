import pandas as pd
import pathlib
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class VacancyRetriever:
    def __init__(self,
                 data_path: str = None,
                 model_name: str = "cointegrated/rubert-tiny2",
                 # Параметры коллекции нам больше не нужны, но оставим для совместимости
                 collection_name: str = "vacancies_mvp",
                 force_reindex: bool = False):

        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.index_path = self.root / "dataset" / "vector_index.pkl"

        self.model = SentenceTransformer(model_name)

        self.index = None
        self.vacancies = []

        # Логика загрузки:
        # Если есть сохраненный индекс и force=False -> грузим.
        # Иначе -> строим заново.

        if not force_reindex and self.index_path.exists():
            print(f"✅ Loading vector index from {self.index_path}...")
            with open(self.index_path, "rb") as f:
                saved_data = pickle.load(f)
                self.index = saved_data["index"]
                self.vacancies = saved_data["vacancies"]
        elif data_path:
            print("⚙️ Building new vector index (sklearn)...")
            self._build_index(data_path)
        else:
            print("⚠️ No index found and no data path provided.")

    def _build_index(self, data_path: str):
        print(f"📥 Loading data from {data_path}...")
        df = pd.read_parquet(data_path)

        # Берем только лучших
        top_df = df[df['is_top_performer'] == True].copy().reset_index(drop=True)
        print(f"   Vectorizing {len(top_df)} vacancies...")

        # 1. Векторизация
        vectors = self.model.encode(top_df['text_clean'].tolist(), show_progress_bar=True)

        # 2. Строим индекс (Brute force для точности, Metric=Cosine)
        # Cosine distance = 1 - Cosine Similarity
        index = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
        index.fit(vectors)

        # 3. Сохраняем данные (нам нужны сами тексты, чтобы их возвращать)
        self.index = index
        self.vacancies = top_df.to_dict("records")

        # 4. Пишем на диск
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "vacancies": self.vacancies
            }, f)

        print("✅ Index built and saved!")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        if not self.index:
            return []

        # Векторизуем запрос
        query_vector = self.model.encode([query])

        # Ищем (возвращает distances и indices)
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=limit)

        results = []
        for i, idx in enumerate(indices[0]):
            # distance - это косинусное расстояние (0..2).
            # Превращаем в similarity (1..-1) для красоты
            score = 1 - distances[0][i]
            vac = self.vacancies[idx]

            results.append({
                "title": vac['vacancy_title'],
                "velocity": vac['velocity'],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    data_file = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset" / "vacancies_processed.parquet"
    retriever = VacancyRetriever(data_path=str(data_file))
    print(retriever.search("Python"))
