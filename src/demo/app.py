import streamlit as st
import requests
import os

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(
    page_title="Job Optimizer AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 30px; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Инициализация памяти ---
if "title" not in st.session_state:
    st.session_state["title"] = ""
if "text" not in st.session_state:
    st.session_state["text"] = ""

# --- Хедер ---
st.markdown('<div class="main-header">🚀 Job Optimizer AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Превратите обычное описание вакансии в магнит для талантов</div>',
            unsafe_allow_html=True)

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("📝 Ввод данных")
    st.caption("Быстрый старт:")
    col_btn1, col_btn2 = st.columns(2)

    if col_btn1.button("Пример: Python"):
        st.session_state["title"] = "Middle Python Developer"
        st.session_state["text"] = "Ищем питониста. Надо знать джанго, sql и докер. Зп по рынку. Работа в офисе."
        st.rerun()

    if col_btn2.button("Пример: Sales"):
        st.session_state["title"] = "Менеджер по продажам"
        st.session_state["text"] = "Нужен продажник. Холодные звонки, встречи. Опыт от 1 года. Оклад + %."
        st.rerun()

    with st.form("input_form"):
        title_val = st.text_input("Название должности", value=st.session_state["title"],
                                  placeholder="Например: Product Manager")
        text_val = st.text_area("Текст вакансии", value=st.session_state["text"], height=300,
                                placeholder="Вставьте описание...")
        submitted = st.form_submit_button("✨ Улучшить вакансию", type="primary")

# --- ЛОГИКА ---
if submitted:
    st.session_state["title"] = title_val
    st.session_state["text"] = text_val

    if not title_val or not text_val:
        st.error("Пожалуйста, заполните оба поля в боковом меню!")
    else:
        with st.spinner("🧠 Нейросеть анализирует рынок и переписывает текст..."):
            try:
                payload = {
                    "vacancies": [{"input_id": "demo", "title": title_val, "text": text_val}]
                }
                response = requests.post(f"{API_URL}/rewrite-batch", json=payload, timeout=120)

                if response.status_code == 200:
                    data = response.json()

                    # ПРОВЕРКА НА ПУСТОЙ ОТВЕТ (Защита от ошибки index out of range)
                    if not data.get("results"):
                        st.error("⚠️ Сервер вернул пустой результат.")
                        st.info("Это значит, что внутри API произошла ошибка. Проверьте терминал сервера.")
                    else:
                        res = data["results"][0]

                        # Метрики
                        score = res['quality_score']
                        color = "green" if score > 80 else "orange" if score > 50 else "red"

                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Quality Score</h3>
                                <h1 style="color: {color};">{score}/100</h1>
                            </div>
                            """, unsafe_allow_html=True)

                        st.divider()

                        col_left, col_right = st.columns(2)

                        with col_left:
                            st.subheader("🔍 Анализ проблем")
                            if res["issues"]:
                                for issue in res["issues"]:
                                    st.warning(f"❌ {issue}")
                            else:
                                st.success("Критических проблем не найдено!")

                            st.subheader("💡 Что улучшено")
                            for note in res["rewrite_notes"]:
                                st.info(f"✅ {note}")

                        with col_right:
                            st.subheader("✨ Готовый текст")
                            st.text_area("Скопируйте результат:", value=res["rewritten_text"], height=600)

                            st.download_button(
                                label="📥 Скачать (.txt)",
                                data=res["rewritten_text"],
                                file_name="vacancy_optimized.txt",
                                mime="text/plain"
                            )

                        with st.expander("🔧 Технические детали"):
                            st.json(res.get("debug", {}))

                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error(f"Ошибка соединения с API: {e}")

else:
    if not st.session_state["title"]:
        st.info("👈 Нажмите на пример слева или введите свой текст.")
