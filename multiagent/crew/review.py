from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.types.usage_metrics import UsageMetrics

from multiagent.crew.agents import build_agents, make_llm
from multiagent.crew.prompts import REPORT_FORMAT, REVIEW_TASK
from multiagent.guardrails import UNVERIFIED_SECTION, Facts, report_guardrail, schema_guardrail, schema_hint
from multiagent.schemas import ReviewResult


def _kickoff(agent: Agent, task: Task, verbose: bool) -> tuple[str, UsageMetrics]:
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose).kickoff()
    return result.raw, result.token_usage


def review_loop(
    model: str | None,
    report_path: str,
    evidence: str,
    facts: Facts,
    request: str | None,
    rounds: int,
    verbose: bool,
) -> UsageMetrics:
    """Let the reviewer criticise the report and the writer fix it, until approved or out of rounds.

    The number of rounds depends on the reviewer's verdict, so this is a feedback loop between agents,
    not a fixed step. Returns the token usage of all review and revision calls.
    """
    usage = UsageMetrics()
    for round_no in range(1, rounds + 1):
        agents = build_agents(make_llm(model), verbose=verbose)
        report = Path(report_path).read_text(encoding="utf-8").split(UNVERIFIED_SECTION)[0].rstrip()
        question = f"Запрос пользователя: «{request}»\n\n" if request else ""

        review = Task(
            description=(
                f"{REVIEW_TASK}\n\n{question}"
                f"<evidence>\n{evidence}\n</evidence>\n\n<report>\n{report}\n</report>"
            ),
            expected_output=f"Только JSON без пояснений по схеме: {schema_hint(ReviewResult)}",
            agent=agents["reviewer"],
            guardrail=schema_guardrail(ReviewResult),
            guardrail_max_retries=2,
        )
        raw, used = _kickoff(agents["reviewer"], review, verbose)
        usage.add_usage_metrics(used)
        verdict = ReviewResult.model_validate_json(raw)

        if verdict.approved or not verdict.issues:
            print(f"[review] круг {round_no}: отчёт принят")
            break
        print(f"[review] круг {round_no}: замечаний {len(verdict.issues)}")
        for issue in verdict.issues:
            print(f"[review]   - {issue}")

        issues = "\n".join(f"- {issue}" for issue in verdict.issues)
        revise = Task(
            description=(
                "Исправь отчёт с учётом замечаний рецензента. Меняй только то, что требуется замечаниями, "
                "и не добавляй фактов, которых нет в данных аналитиков. Верни отчёт целиком.\n\n"
                f"{question}<issues>\n{issues}\n</issues>\n\n"
                f"<evidence>\n{evidence}\n</evidence>\n\n<report>\n{report}\n</report>"
            ),
            expected_output=REPORT_FORMAT,
            agent=agents["report"],
            output_file=report_path,
            guardrail=report_guardrail(facts),
            guardrail_max_retries=2,
        )
        _, used = _kickoff(agents["report"], revise, verbose)
        usage.add_usage_metrics(used)
    return usage
