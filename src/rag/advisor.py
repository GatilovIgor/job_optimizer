from typing import List, Dict
import re
import html
import time
from src.rag.llm import LocalLLM
from src.api.models import VacancyIn, VacancyOut


class VacancyAdvisor:
    def __init__(self):
        print("🔧 Initializing Advisor (Parser Mode)...", flush=True)
        self.llm = LocalLLM()

    def _clean_html(self, raw_text: str) -> str:
        if not raw_text: return ""
        text = html.unescape(raw_text)
        # Убираем только совсем мусор, HTML теги оставляем для скоринга
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
        return text.strip()

    def _analyze_quality(self, text: str) -> Dict:
        """Анализ качества (0-100)"""
        score = 0
        issues = []
        text_lower = text.lower()

        if len(text) < 50:
            return {"score": 0, "issues": ["Текст отсутствует"]}

        # 1. ОБЪЕМ
        if len(text) < 300:
            issues.append("❌ Критически мало текста")
        elif len(text) > 800:
            score += 20
        else:
            score += 10

        # 2. СТРУКТУРА (Самое важное)
        blocks_found = 0
        if "обязанност" in text_lower or "задачи" in text_lower:
            score += 15;
            blocks_found += 1
        else:
            issues.append("❓ Нет блока 'Обязанности'")

        if "требован" in text_lower or "ищем" in text_lower:
            score += 15;
            blocks_found += 1
        else:
            issues.append("❓ Нет блока 'Требования'")

        if "условия" in text_lower or "предлагаем" in text_lower:
            score += 15;
            blocks_found += 1
        else:
            issues.append("❓ Нет блока 'Условия'")

        # БОНУС за полную структуру
        if blocks_found == 3: score += 10

        # 3. ДЕТАЛИ
        money_words = ["руб", "₽", "оклад", "доход", "зарплат", "на руки"]
        if any(w in text_lower for w in money_words):
            score += 10
        else:
            issues.append("💰 Не указана зарплата")

        if any(w in text_lower for w in ["график", "5/2", "2/2", "удален"]):
            score += 10
        else:
            issues.append("📅 Не указан график")

        # 4. ОФОРМЛЕНИЕ
        if "<ul>" in text or "<li>" in text or "•" in text:
            score += 10
        else:
            issues.append("📄 Нет списков")

        return {"score": min(score, 100), "issues": issues}

    def _parse_llm_response(self, raw_text: str, original_title: str) -> Dict:
        """Парсит неструктурированный ответ LLM"""
        result = {
            "title": original_title,
            "specialization": "Не определено",
            "text": raw_text,
            "notes": ["Текст сгенерирован"]
        }

        # Попытка найти заголовок
        title_match = re.search(r'ЗАГОЛОВОК:\s*(.+)', raw_text, re.IGNORECASE)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # Попытка найти сферу
        spec_match = re.search(r'СФЕРА:\s*(.+)', raw_text, re.IGNORECASE)
        if spec_match:
            result["specialization"] = spec_match.group(1).strip()

        # Попытка найти тело описания
        # Ищем всё, что идет после слова "ОПИСАНИЕ:" или просто берем текст, если меток нет
        desc_match = re.split(r'ОПИСАНИЕ:', raw_text, flags=re.IGNORECASE)
        if len(desc_match) > 1:
            # Берем вторую часть (само описание)
            clean_body = desc_match[1].strip()
            # Убираем артефакты Markdown
            clean_body = clean_body.replace("```html", "").replace("```", "")
            result["text"] = clean_body
        else:
            # Если метки нет, просто чистим от заголовков в начале
            clean_body = re.sub(r'ЗАГОЛОВОК:.*\n', '', raw_text)
            clean_body = re.sub(r'СФЕРА:.*\n', '', clean_body)
            result["text"] = clean_body.strip()

        return result

    def process_single_vacancy(self, vac_input: VacancyIn, retriever) -> VacancyOut:
        print(f"▶️ Start processing: {vac_input.input_id}", flush=True)
        start_time = time.time()

        in_title = vac_input.title.strip() if vac_input.title else ""
        raw_text = vac_input.text if vac_input.text else ""

        # 1. Анализ ИСХОДНИКА
        original_analysis = self._analyze_quality(raw_text)

        # 2. Поиск RAG
        search_query = f"{in_title} {raw_text[:200]}"
        references = retriever.search(search_query, limit=1) if retriever else []

        # 3. LLM Генерация (Текстовый режим)
        llm_out = self.llm.generate_rewrite(
            user_vacancy={"title": in_title, "text": raw_text},
            references=references,
            issues=original_analysis["issues"]
        )

        # 4. Парсинг ответа
        parsed = self._parse_llm_response(llm_out["raw_response"], in_title)

        final_text = parsed["text"]
        final_title = parsed["title"]
        final_spec = parsed["specialization"]

        # 5. Анализ РЕЗУЛЬТАТА
        final_analysis = self._analyze_quality(final_text)
        final_score = final_analysis["score"]

        # Искусственный буст, если текст реально длинный и красивый
        if len(final_text) > 1000 and final_score > 80:
            final_score = min(final_score + 10, 100)

        return VacancyOut(
            input_id=vac_input.input_id,
            rewritten_title=final_title,
            rewritten_specialization=final_spec,
            rewritten_text=final_text,
            rewrite_notes=parsed["notes"],
            issues=original_analysis["issues"],  # Показываем старые проблемы
            quality_score=int(final_score),
            original_score=int(original_analysis["score"]),
            safety_flags=[],
            low_confidence_retrieval=(len(references) == 0),
            debug={"processing_time": round(time.time() - start_time, 2)}
        )
