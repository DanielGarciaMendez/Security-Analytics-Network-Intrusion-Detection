"""Render the Random Forest AUROC comparison from the final checker log."""

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


OUTPUT = Path(__file__).resolve().parents[1] / "results/final_checkpoint"
DATASETS = ("UNSW-NB15 official test", "CICIDS2017", "TON_IoT")
TITLE = "Random Forest AUROC across test datasets"
IMAGE = "auroc_comparison.png"


@dataclass(frozen=True)
class Cell:
    name: str
    rows: int
    positive_percent: str
    score: str
    lower: str
    upper: str


def read_cells(log):
    metrics, separator, _ = log.partition("\nResults table: AUROC [95% interval]")
    if not separator:
        raise ValueError("Expected the completed final checker's AUROC results table")
    blocks = re.split(r"^Value: (.+?) +Column: (.+)$", metrics, flags=re.MULTILINE)
    cells = {}
    for name, column, body in zip(blocks[1::3], blocks[2::3], blocks[3::3]):
        rows = re.search(r"^  test rows +(\d+)$", body, re.MULTILINE)
        positives = re.search(r"^  positive rows +\d+ \((\d+\.\d+%)\)$", body, re.MULTILINE)
        auroc = re.search(
            r"^  AUROC +(\d\.\d{4}) \[(\d\.\d{4}), (\d\.\d{4})\]$",
            body,
            re.MULTILINE,
        )
        if column != "Random Forest" or name in cells or not (rows and positives and auroc):
            raise ValueError(f"Unexpected or incomplete final checker cell: {name} / {column}")
        score, lower, upper = auroc.groups()
        if not 0 <= float(lower) <= float(score) <= float(upper) <= 1:
            raise ValueError(f"Invalid AUROC interval for {name}")
        cells[name] = Cell(name, int(rows[1]), positives[1], score, lower, upper)
    if cells.keys() != set(DATASETS):
        raise ValueError("Expected exactly the three test datasets")
    return [cells[name] for name in DATASETS]


def plot(cells):
    fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
    for index, cell in enumerate(cells):
        score, lower, upper = map(float, (cell.score, cell.lower, cell.upper))
        color = "#173e68" if index == 0 else "#247b76"
        ax.errorbar(index, score, yerr=[[score - lower], [upper - score]],
                    fmt="o", color=color, markersize=6, elinewidth=1.5, capsize=5, zorder=3)
        ax.annotate(
            f"{cell.score}\n[{cell.lower}, {cell.upper}]",
            (index, upper), xytext=(0, 11), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color="#18283b", linespacing=1.25,
        )
    ax.axhline(0.5, color="#758392", linestyle="--", linewidth=1, label="Random scoring (0.5)")
    ax.set(xlim=(-0.49, 2.49), ylim=(0, 1),
           ylabel="AUROC", xlabel="Test dataset")
    ax.set_title(TITLE, loc="left", fontsize=15, fontweight="bold", pad=11)
    ax.set_xticks(range(len(cells)), ["UNSW-NB15\nofficial test", "CICIDS2017", "TON_IoT"])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#e2e7ed", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#a1adb9")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(OUTPUT / IMAGE, dpi=300, facecolor="white")
    plt.close(fig)


def write_caption(cells):
    baseline, cicids, ton = cells
    descriptions = [f"{c.name} ({c.rows:,} test rows, {c.positive_percent} positive)" for c in cells]
    caption = (
        f"**Figure.** AUROC for the Random Forest column on {descriptions[0]}, "
        f"{descriptions[1]}, and {descriptions[2]}. Points show AUROC, and vertical "
        "bars show the final checkpoint checker's 95% bootstrap confidence intervals."
    )
    finding = (
        f"Random Forest AUROC fell from {baseline.score} on {baseline.name} to "
        f"{cicids.score} on {cicids.name} and {ton.score} on {ton.name}, indicating "
        "substantially weaker attack discrimination on both external datasets."
    )
    text = f"{caption}\n\n{finding}\n"
    (OUTPUT / "figure_caption.md").write_text(text, encoding="utf-8")
    (OUTPUT / "figure_summary.md").write_text(
        f"# {TITLE}\n\n![{TITLE}]({IMAGE})\n\n{text}", encoding="utf-8",
    )


def main():
    cells = read_cells((OUTPUT / "final_checker_output.txt").read_text(encoding="utf-8"))
    plot(cells)
    write_caption(cells)


if __name__ == "__main__":
    main()
