import re

from crewai.tools import tool
from matplotlib.figure import Figure

from multiagent.config import CHARTS_DIR
from multiagent.tools.tables import _load, _missing_column, _outlier_mask


@tool("make_chart")
def make_chart(
    file_name: str,
    group_by: str,
    value_column: str,
    agg: str = "sum",
    exclude_outliers: bool = False,
    sheet: str = "",
) -> str:
    """Draw a bar chart of agg (sum, mean or count) of `value_column` per `group_by`.
    Set exclude_outliers to drop 1.5*IQR outliers first. Returns the PNG path relative to the report."""
    if agg not in {"sum", "mean", "count"}:
        return "agg должен быть одним из: sum, mean, count"
    df = _load(file_name, sheet)
    if error := _missing_column(df, group_by, value_column):
        return error
    if exclude_outliers:
        df = df[~_outlier_mask(df[value_column])]
    data = df.groupby(group_by)[value_column].agg(agg).sort_values(ascending=False).head(15)

    parts = [file_name.rsplit(".", 1)[0], sheet, group_by, value_column, agg, "clean" if exclude_outliers else ""]
    name = re.sub(r"[^\w]+", "_", "_".join(p for p in parts if p)).strip("_") + ".png"
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    data.plot.bar(ax=ax, color="#4c78a8")
    ax.set_title(f"{agg}({value_column}) по {group_by}" + (" без выбросов" if exclude_outliers else ""))
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=120)
    return f"charts/{name}"
