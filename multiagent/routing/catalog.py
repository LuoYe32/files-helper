import pandas as pd
from crewai.tools import tool

from multiagent.config import DATA_DIR, TABLE_SUFFIXES, TEXT_SUFFIXES
from multiagent.tools.files import extract_text

PREVIEW_CHARS = 400


def _describe_frame(df: pd.DataFrame) -> str:
    columns = ", ".join(map(str, list(df.columns)[:15]))
    return f"{len(df)} строк; колонки: {columns}"


def _preview(path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = " ".join(extract_text(path).split())
        return f"документ; начало: {text[:PREVIEW_CHARS]}"
    if suffix == ".csv":
        return f"таблица; {_describe_frame(pd.read_csv(path))}"
    sheets = pd.read_excel(path, sheet_name=None)
    parts = [f"лист «{name}»: {_describe_frame(df)}" for name, df in sheets.items()]
    return "таблица; " + " || ".join(parts)


def build_catalog() -> str:
    """One short card per file in data/ (type, sheets/columns or text preview) for the router."""
    lines = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES | TABLE_SUFFIXES:
            lines.append(f"- {path.name}: {_preview(path)}")
    return "\n".join(lines)


@tool("file_catalog")
def file_catalog() -> str:
    """Cards of all files in data/: type, text preview or sheets and columns. Use it to choose files."""
    return build_catalog()
