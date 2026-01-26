import pandas as pd
import pathlib
import hashlib
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class VacancyRetriever:
    def __init__(self,
                 data_path: str = None,
                 model_name: str = "cointegrated/rubert-tiny2",
                 collection_name: str = "vacancies_mvp",
                 force_reindex: bool = False):

        self.model_name = model_name
        self.collection_name = collection_name

        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.db_path = self.root / "dataset" / "qdrant_db"

        self.hash_file = self.root / "dataset" / "vacancies.hash"

        self.model = SentenceTransformer(model_name)
        print(f"🔌 Connecting to Vector DB at: {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))

        # --- DEBUG CHECK ---
        if not hasattr(self.client, "search"):
            print("❌ CRITICAL ERROR: QdrantClient has no 'search' method!")
            print(f"   Available methods: {[m for m in dir(self.client) if not m.startswith('_')][:10]}...")
        else:
            print("✅ QdrantClient initialized successfully (search method found).")
        # -------------------

        should_index = False

        if force_reindex:
            print("⚠️ Force reindex requested.")
            should_index = True
        elif not self._check_collection_exists():
            print("⚙️ New database. Indexing needed.")
            should_index = True
        elif data_path and self._has_content_changed(data_path):
            print("🔄 Content changed! Re-indexing top performers...")
            should_index = True

        if should_index and data_path:
            self._index_data(data_path)
        else:
            print("✅ Data is identical to DB. Loaded from disk (Fast start).")

    def _check_collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _calculate_file_hash(self, filepath: str) -> str:
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _has_content_changed(self, data_path: str) -> bool:
        if not self.hash_file.exists():
            return True
        current_hash = self._calculate_file_hash(data_path)
        with open(self.hash_file, 'r') as f:
            stored_hash = f.read().strip()
        return current_hash != stored_hash

    def _save_new_hash(self, data_path: str):
        current_hash = self._calculate_file_hash(data_path)
        with open(self.hash_file, 'w') as f:
            f.write(current_hash)

    def _index_data(self, data_path: str):
        print(f"📥 Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        top_df = df[df['is_top_performer'] == True].copy()
        print(f"   Indexing {len(top_df)} top performers...")

        vectors = self.model.encode(top_df['text_clean'].tolist(), show_progress_bar=True)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )

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
        self._save_new_hash(data_path)
        print("✅ Indexing complete and saved to disk!")

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


if __name__ == "__main__":
    data_file = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset" / "vacancies_processed.parquet"
    retriever = VacancyRetriever(data_path=str(data_file))
