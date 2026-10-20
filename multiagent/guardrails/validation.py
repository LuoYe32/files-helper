import json
import re

from crewai import TaskOutput
from pydantic import BaseModel, ValidationError

from multiagent.config import DATA_DIR
from multiagent.guardrails.facts import Facts
from multiagent.routing.selection import Selection
from multiagent.schemas import RouteResult


def schema_hint(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


def _parse(model: type[BaseModel], raw: str) -> tuple[BaseModel | None, str]:
    """Parse the agent's JSON answer; returns (model, "") or (None, retry message for the agent)."""
    try:
        return model.model_validate_json(_strip_fences(raw)), ""
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
        return None, f"Ответ не соответствует схеме: {problems}. Верни только исправленный JSON."
    except ValueError as e:
        return None, f"Ответ — не валидный JSON ({e}). Верни только JSON без пояснений."


def json_guardrail(model: type[BaseModel], suffixes: set[str], facts: Facts, selection: Selection):
    """Require valid JSON matching `model` that covers every selected data file with one of `suffixes`.

    Validation is done here instead of via output_pydantic: CrewAI then forces tool_choice,
    which Sonnet 5.5 and Opus 5.5 reject with HTTP 400.
    """

    def expected_files() -> set[str]:
        return selection.restrict({p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in suffixes})

    def check(output: TaskOutput):
        parsed, error = _parse(model, output.raw)
        if parsed is None:
            return False, error

        covered = {item.file for item in next(iter(parsed.__dict__.values()))}
        missing = expected_files() - covered
        if missing:
            return False, f"В результате нет файлов: {sorted(missing)}. Обработай их и верни полный JSON."

        result = parsed.model_dump_json(indent=2)
        facts.add(result)
        return True, result

    return check


def schema_guardrail(model: type[BaseModel]):
    """Require valid JSON matching `model`."""

    def check(output: TaskOutput):
        parsed, error = _parse(model, output.raw)
        if parsed is None:
            return False, error
        return True, parsed.model_dump_json(indent=2)

    return check


def route_guardrail(selection: Selection):
    """Require a valid file selection that names only existing files; stores it in `selection`."""

    def check(output: TaskOutput):
        parsed, error = _parse(RouteResult, output.raw)
        if parsed is None:
            return False, error

        existing = {p.name for p in DATA_DIR.iterdir() if p.is_file()}
        unknown = {c.file for c in parsed.files} - existing
        if unknown:
            return False, f"Этих файлов нет в каталоге: {sorted(unknown)}. Выбирай только из каталога."

        selection.files = {c.file for c in parsed.files}
        for choice in parsed.files:
            print(f"[router] {choice.file}: {choice.reason}")
        return True, parsed.model_dump_json(indent=2)

    return check
