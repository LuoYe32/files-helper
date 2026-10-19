from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data").resolve()
OUTPUT_DIR = (ROOT / "output").resolve()
CHARTS_DIR = OUTPUT_DIR / "charts"

MAX_CHARS = 20_000
TEXT_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}

DEFAULT_MODEL = "anthropic/claude-haiku-5-5"

PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}
