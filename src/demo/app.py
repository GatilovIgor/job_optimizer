import streamlit as st
import requests
import os
import html

st.set_page_config(page_title="Job Optimizer AI", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
    .result-box {
        background: white; color: black; padding: 25px; 
        border-radius: 10px; border: 1px solid #ddd; line-height: 1.6;
    }
    /* Делаем метрику крупной */
    [data-testid="stMetricValue"] { font-size: 3rem !important; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- STATE ---
if "title_input" not in st.session_state: st.session_state.title_input = ""
if "text_input" not in st.session_state: st.session_state.text_input = ""
if "spec_input" not in st.session_state: st.session_state.spec_input = ""

st.markdown('<div class="main-header">🚀 Job Optimizer AI</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Ввод данных")
    if st.button("Пример: Продажи"):
        st.session_state.title_input = "Менеджер по продажам"
        st.session_state.text_input = "Требуется активный сотрудник. Холодные звонки, работа с базой. График 5/2. Оклад + процент."
        st.session_state.spec_input = "Продажи"
        st.rerun()

    with st.form("input_form"):
        title_val = st.text_input("Должность", value=st.session_state.title_input)
        spec_val = st.text_input("Сфера (IT, Ритейл...)", value=st.session_state.spec_input)
        text_val = st.text_area("Текст вакансии", value=st.session_state.text_input, height=300)
        submitted = st.form_submit_button("✨ Улучшить", type="primary")

# --- MAIN ---
if submitted:
    st.session_state.title_input = title_val
    st.session_state.text_input = text_val
    st.session_state.spec_input = spec_val

    if not title_val or not text_val:
        st.warning("⚠️ Заполните все поля!")
    else:
        with st.spinner("⚡ Анализ и улучшение..."):
            try:
                payload = {
                    "vacancies": [{"input_id": "1", "title": title_val, "specialization": spec_val, "text": text_val}]}
                response = requests.post(f"{API_URL}/rewrite-batch", json=payload, timeout=120)

                if response.status_code == 200:
                    res = response.json()["results"][0]

                    # --- МЕТРИКА РОСТА ---
                    old_score = res.get('original_score', 0)
                    new_score = res.get('quality_score', 0)
                    delta = new_score - old_score

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="Рост качества вакансии",
                            value=f"{new_score}/100",
                            delta=f"+{delta} баллов" if delta > 0 else f"{delta}",
                            delta_color="normal"
                        )
                        st.caption(f"Было: {old_score}/100 ➔ Стало: {new_score}/100")

                    st.divider()

                    # --- РЕЗУЛЬТАТЫ ---
                    c_left, c_right = st.columns(2)

                    with c_left:
                        st.subheader("🔍 Проблемы исходника")
                        if res.get("issues"):
                            for issue in res["issues"]: st.warning(f"• {issue}")
                        else:
                            st.success("Исходный текст был неплох!")

                    with c_right:
                        st.subheader("✨ Готовый результат")
                        safe_html = html.unescape(res["rewritten_text"])
                        st.markdown(f'<div class="result-box">{safe_html}</div>', unsafe_allow_html=True)

                        with st.expander("🛠 Что исправил AI"):
                            for note in res.get("rewrite_notes", []): st.info(f"✅ {note}")

                        st.download_button("📥 Скачать HTML", data=safe_html, file_name="vacancy.html")

                else:
                    st.error("Ошибка сервера API")
            except Exception as e:
                st.error(f"Ошибка: {e}")
