import pandas as pd
import pathlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VacancyRetriever:
    def __init__(self,
                 data_path: str = None,
                 model_name: str = "cointegrated/rubert-tiny2",  # Легкая модель для русского
                 collection_name: str = "vacancies_mvp"):

        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name

        # Используем локальный Qdrant (в памяти), чтобы не поднимать Docker
        # Для продакшена замените на URL сервера
        self.client = QdrantClient(":memory:")

        if data_path:
            self._index_data(data_path)

    def _index_data(self, data_path: str):
        print(f"📥 Loading data from {data_path}...")
        df = pd.read_parquet(data_path)

        # Индексируем только "Чемпионов", чтобы рекомендовать лучшее
        top_df = df[df['is_top_performer'] == True].copy()
        print(f"   Indexing {len(top_df)} top performers...")

        # Векторизация
        vectors = self.model.encode(top_df['text_clean'].tolist(), show_progress_bar=True)

        # Создаем коллекцию
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )

        # Загружаем точки
        points = []
        for i, row in top_df.reset_index().iterrows():
            payload = {
                "title": row['vacancy_title'],
                "velocity": row['velocity']
            }
            points.append(PointStruct(id=i, vector=vectors[i], payload=payload))

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points
        )
        print("✅ Indexing complete!")

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        query_vector = self.model.encode(query).tolist()

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [
            {
                "title": hit.payload['title'],
                "velocity": hit.payload['velocity'],
                "score": hit.score
            }
            for hit in hits
        ]


# --- DEMO RUN ---
if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    data_file = root / "dataset" / "vacancies_processed.parquet"

    # 1. Инициализация и Индексация
    retriever = VacancyRetriever(data_path=str(data_file))

    # 2. Тестовый поиск
    query = "Ищем менеджера по продажам"
    print(f"\n🔍 Searching for: '{query}'...")

    results = retriever.search(query)

    for r in results:
        print(f"   🏆 Found: {r['title']} (Speed: {r['velocity']:.1f}/day, Score: {r['score']:.2f})")
