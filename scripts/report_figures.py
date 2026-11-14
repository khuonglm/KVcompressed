"""Figures for documents/Status report.md, from the confirmatory results and the follow-up summary.

    uv run python scripts/report_figures.py

Reads results/confirm/report/results.json and documents/figures/followup_gqa.json; writes PNGs to
documents/figures/. Descriptive only: every value drawn is a number already in those files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "figures"
CONFIRM = json.loads((ROOT / "results/confirm/report/results.json").read_text())["datasets"]["gqa"]
FOLLOW = json.loads((OUT / "followup_gqa.json").read_text())

# Reference palette (dataviz skill), light mode
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DELTA = 0.10

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": TEXT,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.constrained_layout.use": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def confirm_entry(result: dict) -> tuple[float, float, float, float]:
    low, high = result["ci95"]
    return result["estimate"], low, high, result["p_two_sided"]


def forest(ax, rows: list[tuple[str, tuple]], color=BLUE, band: bool = False, unit: str = "nats") -> None:
    """Point estimates with 95% intervals, top to bottom; zero line; optional ±δ band."""
    labels = [label for label, _ in rows]
    y = list(range(len(rows)))[::-1]
    if band:
        ax.axvspan(-DELTA, DELTA, color=GRID, alpha=0.6, lw=0, label=f"±δ = {DELTA}")
    ax.axvline(0, color=MUTED, lw=1)
    for yi, (_, (est, low, high, p)) in zip(y, rows, strict=True):
        ax.plot([low, high], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot(est, yi, "o", color=color, ms=7, mec=SURFACE, mew=1.5)
        ax.annotate(f"{est:+.3f}  (p {p:.4f})" if p is not None else f"{est:+.3f}", (high, yi),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean difference, {unit} (95% interval)")
    lo = min(r[1][1] for r in rows)
    hi = max(r[1][2] for r in rows)
    span = hi - lo
    ax.set_xlim(min(lo, -DELTA if band else lo, 0) - 0.08 * span, hi + 0.7 * span)


def fig_tests() -> None:
    tests, explore = CONFIRM["tests"], CONFIRM["exploratory"]
    rows = [
        ("H1  S_H(mis)", confirm_entry(tests["S_H^(mis)"])),
        ("H2  T_match", confirm_entry(tests["T_match"])),
        ("H2  T_irr", confirm_entry(tests["T_irr"])),
        ("H3  E_harm(mis)", confirm_entry(tests["E_harm^(mis)"])),
        ("H4  S_Q(mis)", confirm_entry(tests["S_Q^(mis)"])),
        ("I(mis)", confirm_entry(explore["I^(mis)"])),
        ("S_H(match)", confirm_entry(explore["S_H^(match)"])),
        ("S_H(irr)", confirm_entry(explore["S_H^(irr)"])),
        ("S_H(mis), question-span", confirm_entry(explore["S_H^(mis), question-span"])),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    forest(ax, rows, band=True)
    ax.axhline(3.5, color=GRID, lw=1, ls="--")
    ax.text(ax.get_xlim()[0], 3.35, "exploratory below", fontsize=8, color=MUTED, va="top")
    ax.set_title("Pre-registered tests, GQA, PW-attn, K = 144")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_tests.png")


def fig_margins() -> None:
    m = CONFIRM["mean_margins"]
    cells = [
        ("Full cache, question only (Full_Q)", m["full|Q"], BLUE),
        ("PW-attn A: S_Q, question only", m["pw|A"], BLUE),
        ("PW-attn D: S_mis, question only", m["pw|D_mis"], BLUE),
        ("Full cache + misleading text (Full_mis)", m["full|mis"], ORANGE),
        ("PW-attn C: S_Q + misleading text", m["pw|C_mis"], ORANGE),
        ("PW-attn B: S_mis + misleading text", m["pw|B_mis"], ORANGE),
        ("Question-span B: S_mis + misleading text", m["qspan|B_mis"], ORANGE),
        ("Uniform grid + misleading text", m["uniform|mis"], ORANGE),
        ("Blank image, question only (Blind_Q)", m["blind|Q"], MUTED),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = list(range(len(cells)))[::-1]
    ax.barh(y, [c[1] for c in cells], color=[c[2] for c in cells], height=0.6)
    for yi, (_, value, _) in zip(y, cells, strict=True):
        ax.annotate(f"{value:+.2f}", (value, yi), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [c[0] for c in cells])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-1.2, 6.2)
    ax.set_xlabel("Mean margin, nats: log p(true) − log p(alternative)")
    ax.set_title("Mean margin by cell, K = 144")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_margins.png")


def fig_budget() -> None:
    names = [("S_H^(mis)", "S_H: steering cost", BLUE), ("S_Q^(mis)", "S_Q: steered set, clean prompt", AQUA),
             ("E_harm^(mis)", "E_harm: extra compression harm", ORANGE)]
    budgets = [72, 144, 288]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color=MUTED, lw=1)
    for offset, (key, label, color) in zip((-0.06, 0, 0.06), names, strict=True):
        points = []
        for k in budgets:
            if k == 144:
                block = CONFIRM["tests"]
                points.append(confirm_entry(block[key]))
            else:
                est, low, high, p = FOLLOW["budgets"][str(k)][key]
                points.append((est, low, high, p))
        xs = [i + offset for i in range(len(budgets))]
        ax.plot(xs, [p[0] for p in points], "-", color=color, lw=2, label=label)
        ax.vlines(xs, [p[1] for p in points], [p[2] for p in points], color=color, lw=2)
        ax.plot(xs, [p[0] for p in points], "o", color=color, ms=7, mec=SURFACE, mew=1.5)
    ax.set_xticks(range(len(budgets)), [f"K = {k}\n({k / 576:.1%} kept)" for k in budgets])
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylabel("Mean, nats (95% interval)")
    ax.set_title("Estimands by budget, PW-attn (K = 144 from the confirmatory run)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_budget.png")


def fig_selectors() -> None:
    explore = CONFIRM["exploratory"]
    steer = [("PW-attn (primary)", confirm_entry(CONFIRM["tests"]["S_H^(mis)"])),
             ("Question-span", confirm_entry(explore["S_H^(mis), question-span"])),
             ("LT-early (follow-up)", tuple(FOLLOW["lt"]["S_H^(mis)"]))]
    mitig = CONFIRM["mitigation"]
    rand = FOLLOW["random"]["PW B_mis − random (mis)"]
    quality = [("Random sets (seeds 0–4)", (-rand[0], -rand[2], -rand[1], rand[3])),
               ("Uniform grid", confirm_entry(mitig["uniform (mis)"])),
               ("LT-early", tuple(FOLLOW["lt"]["LT B_mis − PW B_mis"]))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    forest(a, steer, color=BLUE)
    a.set_title("Steering cost S_H(mis) by selector")
    forest(b, quality, color=ORANGE)
    b.set_title("Selector − PW-attn, B_mis margin")
    b.set_xlabel("Difference, nats (95% interval)")
    fig.tight_layout()
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT / "fig4_selectors.png")


def fig_behaviour() -> None:
    rates = CONFIRM["behaviour_rates"]
    fb = FOLLOW["behaviour"]

    def triple(run):
        r = rates[run]
        return r["correct"], r["alt"], r["other"]

    rows = [
        ("Full cache, question only", triple("full|Q")),
        ("Full cache + misleading text", triple("full|mis")),
        ("PW-attn K=288, B (S_mis)", tuple(fb["pw@288|B_mis"])),
        ("PW-attn K=144, B (S_mis)", triple("pw|B_mis")),
        ("PW-attn K=144, C (S_Q)", triple("pw|C_mis")),
        ("PW-attn K=72, B (S_mis)", tuple(fb["pw@72|B_mis"])),
        ("PW-attn K=72, C (S_Q)", tuple(fb["pw@72|C_mis"])),
        ("Question-span K=144, B", triple("qspan|B_mis")),
        ("LT-early K=144, B", tuple(fb["lt@144|B_mis"])),
        ("Uniform K=144", triple("uniform|mis")),
        ("Random K=144 (mean of 5 seeds)", tuple(fb["random|mis (mean of seeds 0-4)"])),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = list(range(len(rows)))[::-1]
    left = [0.0] * len(rows)
    for index, (name, color) in enumerate((("Correct", BLUE), ("Misleading alternative", ORANGE), ("Other", GRID))):
        values = [r[1][index] for r in rows]
        ax.barh(y, values, left=left, color=color, height=0.6, label=name, edgecolor=SURFACE, linewidth=1)
        if index == 0:
            for yi, v in zip(y, values, strict=True):
                ax.text(v - 0.01, yi, f"{v:.0%}", ha="right", va="center", fontsize=8, color=SURFACE, fontweight="bold")
        left = [a + v for a, v in zip(left, values, strict=True)]
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of greedy answers")
    ax.set_title("Greedy answers, misleading text in all rows but the first")
    ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_behaviour.png")


def fig_mechanism() -> None:
    h6 = [(f"{rule} rule", confirm_entry(CONFIRM["H6"][rule])) for rule in ("primary", "fallback", "center", "expanded")]
    restore = FOLLOW["restore"]["4"]
    rest = [("Evidence − non-evidence (EQ3)", tuple(restore["treatment − control (EQ3)"])),
            ("Evidence restored − none", tuple(restore["treatment − reference"])),
            ("Non-evidence restored − none", tuple(restore["control − reference"]))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    forest(a, h6, color=AQUA, unit="share of evidence region")
    a.set_title("H6: evidence kept, S_Q − S_mis")
    forest(b, rest, color=VIOLET)
    b.set_title("Restoration, m = 4 (49 items)")
    fig.tight_layout()
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT / "fig6_mechanism.png")


def fig_mitigation() -> None:
    mitig = CONFIRM["mitigation"]
    rows = [(label, confirm_entry(mitig[key])) for label, key in (
        ("Isolation (C_mis − B_mis)", "isolation (C_mis)"),
        ("Reserve 10% (B_mis+r − B_mis)", "reserve (B_mis + r)"),
        ("Both (C_mis+r − B_mis)", "both (C_mis + r)"),
        ("Reserve, neutral (A+r − A)", "reserve cost, neutral (A + r − A)"),
        ("Reserve, matching (B_match+r − B_match)", "reserve cost, matching (B_match + r − B_match)"),
        ("Isolation, matching (C_match − B_match)", "isolation cost, matching (C_match − B_match)"),
    )]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    forest(ax, rows, color=BLUE)
    ax.set_title("Mitigation and its side effects, K = 144")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_mitigation.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (fig_tests, fig_margins, fig_budget, fig_selectors, fig_behaviour, fig_mechanism, fig_mitigation):
        make()
    print("figures written to", OUT)
