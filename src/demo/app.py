import streamlit as st
import requests
import os
import html

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(page_title="Job Optimizer AI", page_icon="🚀", layout="wide")

# Стили с исправленной видимостью
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #FF4B4B; text-align: center; margin-bottom: 5px; }
    .sub-header { font-size: 1.1rem; color: #ccc; text-align: center; margin-bottom: 30px; }
    .metric-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center; 
        border: 2px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 { color: #333333 !important; margin-bottom: 0; }
    .result-box {
        background: white; 
        color: black; 
        padding: 25px; 
        border-radius: 10px; 
        border: 1px solid #ddd;
        font-family: sans-serif;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

if "title" not in st.session_state: st.session_state["title"] = ""
if "text" not in st.session_state: st.session_state["text"] = ""

st.markdown('<div class="main-header">🚀 Job Optimizer AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Мгновенное превращение текста вакансии в профессиональный оффер</div>',
            unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📝 Ввод данных")
    with st.form("input_form"):
        title_val = st.text_input("Название должности", value=st.session_state["title"])
        text_val = st.text_area("Текст вакансии", value=st.session_state["text"], height=350)
        submitted = st.form_submit_button("✨ Улучшить вакансию", type="primary")

# --- ЛОГИКА ---
if submitted:
    if not title_val or not text_val:
        st.error("Заполните поля!")
    else:
        with st.spinner("🧠 Нейросеть анализирует рынок..."):
            try:
                payload = {"vacancies": [{"input_id": "demo", "title": title_val, "text": text_val}]}
                response = requests.post(f"{API_URL}/rewrite-batch", json=payload, timeout=180)

                if response.status_code == 200:
                    res = response.json()["results"][0]

                    # Score UI
                    score = res.get('quality_score', 0)
                    color = "#28a745" if score > 80 else "#fd7e14" if score > 50 else "#dc3545"

                    col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
                    with col_m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Оценка качества текста</h3>
                            <h1 style="color: {color}; font-size: 4rem; margin: 0;">{score}/100</h1>
                        </div>
                        """, unsafe_allow_html=True)

                    st.divider()

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("🔍 Анализ и улучшения")
                        # Ошибки
                        if res.get("issues"):
                            for issue in res["issues"]:
                                st.warning(f"⚠️ {issue}")
                        # Что сделано (теперь на русском)
                        if res.get("rewrite_notes"):
                            for note in res["rewrite_notes"]:
                                st.info(f"✅ {note}")

                    with col_right:
                        st.subheader("✨ Готовый текст")
                        # Декодируем и рендерим
                        raw_html = html.unescape(res["rewritten_text"])
                        st.markdown(f'<div class="result-box">{raw_html}</div>', unsafe_allow_html=True)

                        st.download_button("📥 Скачать HTML", data=raw_html, file_name="vacancy.html", mime="text/html")

                else:
                    st.error("Ошибка API")
            except Exception as e:
                st.error(f"Ошибка: {e}")
