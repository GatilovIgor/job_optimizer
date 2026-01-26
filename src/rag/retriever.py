import pandas as pd
import pathlib
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class VacancyRetriever:
    def __init__(self,
                 data_path: str = None,
                 model_name: str = "cointegrated/rubert-tiny2",
                 force_reindex: bool = False):

        # Путь к корню проекта (рассчитывается от текущего файла)
        self.root = pathlib.Path(__file__).resolve().parent.parent.parent
        self.index_path = self.root / "dataset" / "vector_index.pkl"

        self.model = SentenceTransformer(model_name)

        self.index = None
        self.vacancies = []

        # Логика загрузки:
        # Если есть сохраненный индекс и force=False -> грузим.
        # Иначе -> строим заново из Parquet.

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

        # 1. Фильтр: берем только успешные вакансии
        if 'is_top_performer' in df.columns:
            top_df = df[df['is_top_performer'] == True].copy().reset_index(drop=True)
            print(f"   Filtering: {len(df)} -> {len(top_df)} top performers.")
        else:
            print("⚠️ 'is_top_performer' column missing. Using all data.")
            top_df = df.copy()

        if len(top_df) == 0:
            print("❌ No vacancies found for indexing!")
            return

        # 2. Формируем "Rich Embedding Context" (Богатый контекст)
        # Вектор должен учитывать профессию и навыки, а не только описание.
        # Формат: "Заголовок. Профиль. Навыки. Текст..."
        top_df['embedding_text'] = (
                top_df['vacancy_title'].fillna('') + ". " +
                top_df['specialization'].fillna('') + ". " +
                top_df['skills_str'].fillna('') + ". " +
                top_df['text_clean'].fillna('')
        )

        # Обрезаем слишком длинные тексты (модель сама обрежет, но лучше заранее)
        top_df['embedding_text'] = top_df['embedding_text'].str.slice(0, 2000)

        print(f"   Vectorizing {len(top_df)} items...")
        vectors = self.model.encode(top_df['embedding_text'].tolist(), show_progress_bar=True)

        # 3. Строим индекс (Brute force + Cosine)
        index = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
        index.fit(vectors)

        self.index = index
        # Сохраняем словарь записей, чтобы возвращать их LLM
        # ВАЖНО: сохраняем 'vacancy_description' (HTML), а не text_clean
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

        # Векторизуем запрос пользователя
        query_vector = self.model.encode([query])

        # Ищем
        distances, indices = self.index.kneighbors(query_vector, n_neighbors=limit)

        results = []
        for i, idx in enumerate(indices[0]):
            vac = self.vacancies[idx]

            # Косинусное расстояние (0..2) -> Сходство (1..-1)
            score = 1 - distances[0][i]

            results.append({
                "title": vac['vacancy_title'],
                # Отдаем LLM исходный HTML для обучения структуре
                "html_text": vac.get('vacancy_description', vac.get('text_clean', '')),
                "velocity": vac.get('velocity', 0.0),
                "score": float(score)
            })

        # Сортировка: сначала самые похожие (score), при равенстве - самые эффективные (velocity)
        results.sort(key=lambda x: (x['score'], x['velocity']), reverse=True)
        return results
