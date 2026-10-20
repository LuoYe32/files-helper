import math
import re

NUMBER_RE = re.compile("\\d{1,3}(?:[  ]\\d{3})+(?:[.,]\\d+)?|\\d+(?:[.,]\\d+)?")


def parse_numbers(text: str) -> list[float]:
    return [float(raw.replace(" ", "").replace(" ", "").replace(",", ".")) for raw in NUMBER_RE.findall(text)]


def _is_significant(raw: str, value: float) -> bool:
    if re.fullmatch(r"(0?[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])", raw):
        return False
    has_fraction = bool(re.search(r"[.,]\d", raw))
    is_year = not has_fraction and 1900 <= value <= 2100
    return not is_year and (has_fraction or value >= 1000)


class Facts:
    """Numbers reported by the analysts; used to verify numbers in the final report."""

    def __init__(self) -> None:
        self.numbers: list[float] = []

    def add(self, text: str) -> None:
        self.numbers.extend(parse_numbers(text))

    def unverified(self, report: str) -> list[str]:
        """Significant numbers in the report that no analyst reported (within 0.5%)."""
        missing = []
        for raw in NUMBER_RE.findall(report):
            value = parse_numbers(raw)[0]
            if not _is_significant(raw, value):
                continue
            if not any(math.isclose(value, known, rel_tol=0.005, abs_tol=0.01) for known in self.numbers):
                missing.append(raw)
        return list(dict.fromkeys(missing))


_active: Facts | None = None


def activate(facts: Facts) -> None:
    """Make `facts` collect the output of every recorded tool until the next call."""
    global _active
    _active = facts


def record(text: str) -> None:
    if _active is not None:
        _active.add(text)
