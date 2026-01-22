import re

def normalize_text(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t)      # простая очистка HTML
    t = re.sub(r"\s+", " ", t).strip()
    return t

def basic_issues(text: str) -> list[str]:
    issues = []
    if len(text) < 300:
        issues.append("Слишком короткое описание — мало конкретики")
    if len(text) > 6000:
        issues.append("Слишком длинный текст — вероятно, тяжело читать")
    # простые эвристики: “зарплата”
    if not re.search(r"\b(₽|руб|руб\.|€|\$|k\b|тыс)\b", text.lower()):
        issues.append("Не найдена зарплатная вилка/упоминание компенсации")
    # “удалёнка”
    if not re.search(r"\b(удаленк|remote|гибрид|hybrid)\b", text.lower()):
        issues.append("Не найдено явное указание формата работы (удалёнка/гибрид/офис)")
    return issues

def heuristic_quality_score(text: str, issues: list[str]) -> int:
    score = 85
    score -= 10 * len(issues)
    score -= 10 if len(text) > 6000 else 0
    score -= 10 if len(text) < 300 else 0
    return max(0, min(100, score))
