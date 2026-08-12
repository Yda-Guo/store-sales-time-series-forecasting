"""Compact reproducible exploratory analysis for the store-sales data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import load_raw_data


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures" / "eda"
REPORT_PATH = ROOT / "reports" / "eda_summary.md"


def _save(series, filename: str, title: str, xlabel: str, ylabel: str, kind: str = "line") -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    series.plot(ax=ax, kind=kind)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / filename, dpi=150); plt.close(fig)


def main() -> None:
    raw = load_raw_data(); train = raw["train"].copy()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    daily = train.groupby("date")["sales"].sum()
    weekday = train.assign(weekday=train["date"].dt.day_name()).groupby("weekday")["sales"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    families = train.groupby("family")["sales"].sum().nlargest(12).sort_values()
    promotion = train.groupby(train["onpromotion"].gt(0))["sales"].mean().rename(index={False: "Not promoted", True: "Promoted"})
    _save(daily, "daily_sales.png", "Total Daily Sales", "Date", "Sales")
    _save(weekday, "weekday_sales.png", "Mean Row-Level Sales by Weekday", "Weekday", "Mean sales", "bar")
    _save(families, "top_families.png", "Largest Product Families by Total Sales", "Family", "Sales", "barh")
    _save(promotion, "promotion_sales.png", "Mean Sales by Promotion Status", "Promotion", "Mean sales", "bar")
    REPORT_PATH.write_text(f"""# Exploratory Data Analysis

The training panel contains {len(train):,} rows from {train['date'].min().date()} through {train['date'].max().date()}, covering {train['store_nbr'].nunique()} stores and {train['family'].nunique()} product families. The figures below describe associations in the observed panel; they do not establish causal effects.

![Daily sales](figures/eda/daily_sales.png)

![Weekday pattern](figures/eda/weekday_sales.png)

![Largest families](figures/eda/top_families.png)

![Promotion association](figures/eda/promotion_sales.png)

The panel has strong scale differences across families and stores, visible weekly structure, and a positive descriptive association between promotion status and sales. Those patterns motivate log-scale error, pooled categorical models, calendar inputs, and known-future promotion features. The forecast protocol still freezes every target-history input at the forecast origin.
""", encoding="utf-8")


if __name__ == "__main__":
    main()

