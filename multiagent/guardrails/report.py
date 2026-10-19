import re
from pathlib import Path

from crewai import TaskOutput

from multiagent.config import OUTPUT_DIR
from multiagent.guardrails.facts import Facts

REQUIRED_HEADINGS = ("## Резюме", "## Документы", "## Таблицы", "## Выводы")


def report_guardrail(facts: Facts):
    """Require the mandatory sections and existing charts; flag numbers not backed by the analysts.

    Unverified numbers do not fail the task: they are listed in an appended warning section.
    """

    def check(output: TaskOutput):
        text = output.raw
        absent = [h for h in REQUIRED_HEADINGS if h not in text]
        if absent:
            return False, f"В отчёте нет разделов: {absent}. Добавь их и верни отчёт целиком."
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if not Path(OUTPUT_DIR, rel).is_file():
                return False, f"Файл графика не найден: {rel}. Используй только пути, которые вернул make_chart."

        unverified = facts.unverified(text)
        if unverified:
            print(f"[fact-check] numbers not found in analyst results: {unverified}")
            text += (
                "\n\n## Непроверенные числа\n\n"
                "Эти числа не найдены в результатах аналитиков, проверьте их вручную: " + ", ".join(unverified)
            )
        return True, text

    return check
