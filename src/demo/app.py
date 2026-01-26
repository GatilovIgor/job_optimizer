import streamlit as st
import requests

st.set_page_config(page_title="Job Optimizer AI", layout="wide", page_icon="🚀")

API_URL = "http://127.0.0.1:8000/rewrite-batch"

# Инициализация состояния
if "title_in" not in st.session_state: st.session_state["title_in"] = ""
if "spec_in" not in st.session_state: st.session_state["spec_in"] = ""
if "text_in" not in st.session_state: st.session_state["text_in"] = ""

st.markdown("""
<style>
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .issue-tag {
        background-color: #422006;
        color: #fcd34d;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #f59e0b;
    }
    .result-box {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 10px;
        margin-top: 5px;
        font-size: 14px;
        line-height: 1.6;
    }
    .label-text {
        color: #9ca3af;
        font-size: 12px;
        margin-bottom: 2px;
        margin-top: 15px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .field-box {
        background-color: #374151;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #4b5563;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📝 Ввод данных")


    def fill_example():
        st.session_state["title_in"] = "Менеджер по продажам"
        st.session_state["spec_in"] = "Продажи"
        st.session_state[
            "text_in"] = "Требуется активный сотрудник. Холодные звонки, работа с базой. График 5/2. Оклад + процент."


    st.button("Пример: Продажи", on_click=fill_example)

    title = st.text_input("Должность", key="title_in")
    spec = st.text_input("Сфера (IT, Ритейл...)", key="spec_in")
    text = st.text_area("Текст вакансии", height=300, key="text_in")

    submit = st.button("✨ Улучшить", type="primary")

st.title("🚀 Job Optimizer AI")

if submit:
    with st.spinner("🤖 ИИ думает..."):
        try:
            payload = {
                "vacancies": [{
                    "input_id": "demo",
                    "title": title,
                    "specialization": spec,
                    "text": text
                }]
            }

            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data["results"][0]

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                diff = result['quality_score'] - result['original_score']
                diff_html = f"<span style='color:#4ade80'>↑ +{diff} баллов</span>" if diff > 0 else ""

                # ИСПРАВЛЕНИЕ: Убраны отступы, чтобы HTML не ломался
                html_code = f"""
<div class="metric-card">
<h3 style="margin:0; color:#9ca3af">Рост качества вакансии</h3>
<h1 style="font-size: 60px; margin:0; color:white">{result['quality_score']}/100</h1>
{diff_html}
<p style="color:#6b7280; font-size:12px; margin-top:5px">Было: {result['original_score']}/100 → Стало: {result['quality_score']}/100</p>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                st.subheader("🔍 Проблемы исходника")
                if not result['issues']:
                    st.success("🎉 Проблем не найдено! Вакансия отличная.")
                else:
                    for issue in result['issues']:
                        st.markdown(f'<div class="issue-tag">• {issue}</div>', unsafe_allow_html=True)

            with c2:
                st.subheader("✨ Готовый результат")

                st.markdown('<div class="label-text">ДОЛЖНОСТЬ</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-box">{result.get("rewritten_title", "Не определено")}</div>',
                            unsafe_allow_html=True)

                st.markdown('<div class="label-text">СФЕРА</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-box">{result.get("rewritten_specialization", "Не определено")}</div>',
                            unsafe_allow_html=True)

                st.markdown('<div class="label-text">ТЕКСТ ВАКАНСИИ</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-box">{result["rewritten_text"]}</div>', unsafe_allow_html=True)

            st.write("")
            with st.expander("🛠 Что исправил AI?"):
                for note in result['rewrite_notes']:
                    st.write(f"- {note}")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
else:
    st.info("👈 Заполните данные и нажмите кнопку")
