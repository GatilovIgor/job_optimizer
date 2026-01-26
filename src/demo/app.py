import streamlit as st
import sys
import pathlib
import time

# Добавляем путь к src, чтобы видеть наши модули
root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))


# Импортируем наши "мозги"
# Используем кэширование Streamlit, чтобы не грузить модель при каждом клике
@st.cache_resource
def load_models():
    print("⏳ Loading Models...")
    from src.rag.retriever import VacancyRetriever
    from src.rag.advisor import VacancyAdvisor

    data_path = root / "dataset" / "vacancies_processed.parquet"
    retriever = VacancyRetriever(data_path=str(data_path))
    advisor = VacancyAdvisor()
    return retriever, advisor


# Заголовок страницы
st.set_page_config(page_title="Job Optimizer AI", layout="wide")
st.title("🚀 Job Optimizer AI")
st.markdown("### Улучшите свою вакансию с помощью ИИ и рыночных данных")

# Боковая панель
with st.sidebar:
    st.info("💡 Эта система ищет успешные вакансии в базе и генерирует советы на локальной нейросети.")

# Загрузка моделей (один раз)
with st.spinner("Загрузка нейросетей (это может занять минуту)..."):
    retriever, advisor = load_models()

# Форма ввода
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Название вакансии", value="Senior Python Developer")
with col2:
    description = st.text_area("Краткое описание (опционально)", height=100)

if st.button("✨ Проанализировать", type="primary"):
    if not title:
        st.error("Введите название вакансии!")
    else:
        # 1. Поиск (RAG)
        query = f"{title} {description}"
        with st.status("🔍 Поиск успешных референсов в базе...", expanded=True) as status:
            champions = retriever.search(query, limit=3)
            status.update(label="✅ Референсы найдены!", state="complete", expanded=False)

        # 2. Показываем найденные вакансии
        st.subheader("📊 Рыночный Benchmark (Топ-3 похожих)")
        cols = st.columns(3)
        for i, (col, vac) in enumerate(zip(cols, champions)):
            with col:
                st.success(f"Score: {vac['score']:.2f}")
                st.write(f"**{vac['title']}**")
                st.caption(f"Velocity: {vac['velocity']:.1f} откл/день")

        # 3. Генерация совета (LLM)
        st.subheader("🤖 AI Рекомендации")
        with st.spinner("Нейросеть пишет анализ (подождите 10-30 сек)..."):
            start_time = time.time()
            analysis = advisor.analyze(query, champions)
            duration = time.time() - start_time

        # Вывод результата
        st.markdown(analysis['ai_advice_text'])
        st.caption(f"Время генерации: {duration:.1f} сек на CPU")
