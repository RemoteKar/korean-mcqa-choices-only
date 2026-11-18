"""Figure 1: ChoicesOnly solvability by question polarity, with 95% bootstrap CIs.

Usage: python make_figure.py [--results results] [--out results/latex/polarity.pdf]
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, slots 1-2 (validated adjacent pair, light surface).
SERIES = {"positive": ("#2a78d6", "o", "Positive question"), "negated": ("#eb6834", "s", "Negated question")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BENCHES = [("KMMLU", "KMMLU"), ("CLIcK exam", "CLIcK exam"), ("CLIcK textbook", "CLIcK textbook"),
           ("Global-MMLU ko", "GMMLU-ko"), ("Global-MMLU en", "GMMLU-en")]


def parse(cell):
    return [float(x) for x in re.match(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", cell).groups()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/latex/polarity.pdf")
    args = parser.parse_args()
    with open(os.path.join(args.results, "tables", "polarity.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), sharey=True)
    for ax, (key, title) in zip(axes, [("qwen", "Qwen2.5-3B-Instruct"), ("exaone", "EXAONE-3.5-2.4B-Instruct")]):
        for i, (bench, label) in enumerate(BENCHES):
            for dy, pol in ((-0.15, "positive"), (0.15, "negated")):
                r = next(r for r in rows if r["model"] == key and r["benchmark"] == bench and r["polarity"] == pol)
                m, lo, hi = parse(r["s_co"])
                color, marker, _ = SERIES[pol]
                y = i + dy
                ax.plot([lo, hi], [y, y], color=color, lw=1.5, solid_capstyle="round", zorder=2)
                ax.plot(m, y, marker=marker, ms=4.5, color=color, mec="white", mew=0.8, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel(r"$s_{\mathrm{CO}}$ (%), 0 = chance", color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(-27, 32)
    shares = {r["benchmark"]: float(r["share"]) for r in rows if r["model"] == "qwen" and r["polarity"] == "negated"}
    axes[0].set_yticks(range(len(BENCHES)))
    axes[0].set_yticklabels([f"{label} ({100 * shares[bench]:.0f}%)" for bench, label in BENCHES])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=mk, lw=1.5, ms=4.5, mec="white", label=lab)
               for c, mk, lab in SERIES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
