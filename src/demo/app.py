import streamlit as st
import requests

# Настройки страницы
st.set_page_config(page_title="Job Optimizer Pro", layout="wide", page_icon="📈")

API_URL = "http://127.0.0.1:8000/optimize"

st.title("📈 Job Optimizer: Peak Efficiency")
st.markdown("Оптимизация вакансий на основе исторических данных эффективности.")

# --- БОКОВАЯ ПАНЕЛЬ: ВВОД 5 ПОЛЕЙ ---
with st.sidebar:
    st.header("📝 Ввод данных")


    def load_example():
        st.session_state['in_profile'] = "Продажи"
        st.session_state['in_city'] = "Москва"
        st.session_state['in_spec'] = "Менеджер по продажам"
        st.session_state['in_title'] = "Менеджер по продажам (холодные звонки)"
        st.session_state['in_desc'] = "Требуется менеджер. Звонки, встречи, CRM. Оклад + %."


    st.button("Загрузить пример", on_click=load_example)

    with st.form("input_form"):
        profile = st.text_input("Профиль", key='in_profile')
        city = st.text_input("Город", key='in_city')
        specialization = st.text_input("Специализация", key='in_spec')
        title = st.text_input("Заголовок вакансии", key='in_title')
        description = st.text_area("Описание вакансии", height=200, key='in_desc')

        submit = st.form_submit_button("🚀 Оптимизировать")

# --- ОСНОВНАЯ ЧАСТЬ: РЕЗУЛЬТАТЫ ---
if submit:
    if not (title and description):
        st.error("Заполните хотя бы Заголовок и Описание!")
    else:
        with st.spinner("🔍 Анализ эффективности и генерация..."):
            payload = {
                "vacancies": [{
                    "input_id": "demo_1",
                    "profile": profile,
                    "city": city,
                    "specialization": specialization,
                    "vacancy_title": title,
                    "vacancy_description": description
                }]
            }

            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
                res = data["results"][0]

                # Отображение улучшений
                st.subheader("✨ Результат оптимизации")

                # Таблица сравнения
                col1, col2 = st.columns(2)

                with col1:
                    st.info("Было")
                    st.text_input("Заголовок (Orig)", title, disabled=True)
                    st.text_area("Описание (Orig)", description, height=300, disabled=True)
                    st.caption(f"Профиль: {profile} | Город: {city} | Спец: {specialization}")

                with col2:
                    st.success("Стало (Optimized)")
                    st.text_input("Заголовок (New)", res['vacancy_title'])
                    st.text_area("Описание (New)", res['vacancy_description'], height=300)
                    st.caption(f"Профиль: {res['profile']} | Город: {res['city']} | Спец: {res['specialization']}")

                # Примечания
                with st.expander("💡 Что улучшил AI?", expanded=True):
                    if res.get('improvement_notes'):
                        for note in res['improvement_notes']:
                            st.write(f"- {note}")
                    else:
                        st.write("Нет примечаний.")

            except Exception as e:
                st.error(f"Ошибка соединения с API: {e}")
                st.warning("Убедитесь, что backend запущен: python -m src.api.main")
else:
    st.info("👈 Заполните форму слева, чтобы начать.")
