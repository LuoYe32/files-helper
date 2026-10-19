import pandas as pd
from crewai.tools import tool

from multiagent.tools.files import _safe_path


def _load(file_name: str, sheet: str = "") -> pd.DataFrame:
    """Load a csv/xlsx table from data/; `sheet` selects an xlsx sheet (first one by default)."""
    path = _safe_path(file_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        names = pd.ExcelFile(path).sheet_names
        if sheet and sheet not in names:
            raise ValueError(f"Нет листа '{sheet}'. Доступны: {names}")
        return pd.read_excel(path, sheet_name=sheet or names[0])
    raise ValueError(f"Неподдерживаемый формат таблицы: {path.suffix}")


def _sheet_names(file_name: str) -> list[str]:
    path = _safe_path(file_name)
    return pd.ExcelFile(path).sheet_names if path.suffix.lower() in {".xlsx", ".xls"} else []


def _outlier_mask(series: pd.Series) -> pd.Series:
    """True where a value lies outside the 1.5*IQR fences."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)


def _missing_column(df: pd.DataFrame, *columns: str) -> str | None:
    for col in columns:
        if col not in df.columns:
            return f"Нет колонки '{col}'. Доступны: {list(df.columns)}"
    return None


@tool("describe_table")
def describe_table(file_name: str, sheet: str = "") -> str:
    """Describe a table: sheets (xlsx), size, column types, first rows, numeric statistics, missing values.
    For xlsx call it once per sheet."""
    df = _load(file_name, sheet)
    sheets = _sheet_names(file_name)
    header = f"Листы файла: {sheets}\nПоказан лист: {sheet or sheets[0]}\n\n" if sheets else ""
    return (
        f"{header}Строк: {len(df)}, колонок: {len(df.columns)}\n\n"
        f"Колонки и типы:\n{df.dtypes.to_string()}\n\n"
        f"Первые строки:\n{df.head(5).to_string()}\n\n"
        f"Статистика:\n{df.describe().to_string()}\n\n"
        f"Пропуски:\n{df.isna().sum().to_string()}"
    )


@tool("group_table")
def group_table(file_name: str, group_by: str, value_column: str, agg: str = "sum", sheet: str = "") -> str:
    """Group by `group_by` and aggregate `value_column` with agg: sum, mean, count, min or max."""
    if agg not in {"sum", "mean", "count", "min", "max"}:
        return "agg должен быть одним из: sum, mean, count, min, max"
    df = _load(file_name, sheet)
    if error := _missing_column(df, group_by, value_column):
        return error
    return df.groupby(group_by)[value_column].agg(agg).sort_values(ascending=False).to_string()


@tool("find_outliers")
def find_outliers(file_name: str, column: str, sheet: str = "") -> str:
    """Find outlier rows in a numeric column using the 1.5*IQR rule."""
    df = _load(file_name, sheet)
    if error := _missing_column(df, column):
        return error
    mask = _outlier_mask(df[column])
    return df[mask].head(30).to_string() + f"\nВсего выбросов: {int(mask.sum())}" if mask.any() else "Выбросов не найдено"


@tool("summary_without_outliers")
def summary_without_outliers(file_name: str, column: str, sheet: str = "") -> str:
    """Sum, mean and median of a numeric column, with all values and without 1.5*IQR outliers."""
    df = _load(file_name, sheet)
    if error := _missing_column(df, column):
        return error
    s = df[column].dropna()
    clean = s[~_outlier_mask(s)]

    def fmt(x: pd.Series) -> str:
        return f"n={len(x)}, сумма={x.sum():.2f}, среднее={x.mean():.2f}, медиана={x.median():.2f}"

    return f"Все значения: {fmt(s)}\nБез выбросов: {fmt(clean)}\nИсключено выбросов: {len(s) - len(clean)}"


@tool("data_quality")
def data_quality(file_name: str, sheet: str = "") -> str:
    """Check data quality: duplicate rows, duplicate values in the first column (likely an id),
    missing values per column and negative numbers."""
    df = _load(file_name, sheet)
    id_col = df.columns[0]
    numeric = df.select_dtypes("number")
    negatives = (numeric < 0).sum()
    lines = [
        f"Полных дубликатов строк: {int(df.duplicated().sum())}",
        f"Повторов значений в первой колонке '{id_col}': {int(df[id_col].duplicated().sum())}",
        f"Пропуски по колонкам:\n{df.isna().sum()[df.isna().sum() > 0].to_string() or 'нет'}",
        f"Отрицательные значения:\n{negatives[negatives > 0].to_string() or 'нет'}",
    ]
    return "\n".join(lines)
