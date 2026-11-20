"""Generate publication-quality SVG figures for meta-rsi and benchmarks/gsb.

Outputs (written to docs/figures/ by default):
  1. fig_outcomes.svg  — Zero-shot outcome breakdown on 480x8 GSB v1 test tasks
  2. fig_mercator.svg  — Web Mercator (EPSG:3857) length error vs latitude
  3. fig_rsi_gates.svg — 4-seed RSI gate comparison (Seed / naive / preserve / metamorphic)

Run:
    nix-shell -p "python3.withPackages(ps:[ps.matplotlib ps.numpy])" --run "python scripts/plot_figures.py"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
COLORS = ["#008300", "#eda100", "#6250d6", "#e34948"]  # correct, lucky, silent_wrong, error
LINE = "#2a78d6"
CATS = ["Correct at all 3 places", "Right at some places only (lucky)", "Runs, wrong everywhere", "Error"]

FALLBACK_ZEROSHOT = [
    ("Gemma 4 E4B", [0.224, 0.269, 0.293, 0.214]),
    ('E4B + "within 1%"', [0.242, 0.220, 0.278, 0.260]),
    ("E4B + 1% + CRS rules", [0.440, 0.003, 0.139, 0.418]),
    ("Gemma 4 26B A4B", [0.717, 0.134, 0.104, 0.045]),
    ("Qwen3.8-27B", [0.622, 0.089, 0.110, 0.178]),
]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "path",
    })


def _load_zeroshot_rows(zdir: Path) -> list[tuple[str, list[float]]]:
    mapping = [
        ("e4b_think", "Gemma 4 E4B"),
        ("e4b_think_hint", 'E4B + "within 1%"'),
        ("e4b_think_rules", "E4B + 1% + CRS rules"),
        ("g26b_think", "Gemma 4 26B A4B"),
        ("q27_low", "Qwen3.8-27B"),
    ]
    rows = []
    for key, label in mapping:
        p = zdir / f"{key}.summary.json"
        if not p.exists():
            return FALLBACK_ZEROSHOT
        s = json.loads(p.read_text())["summary"]["ALL"]
        rows.append((label, [s["correct"], s["lucky"], s["silent_wrong"], s["error"]]))
    return rows


def fig_outcomes(out_dir: Path, zdir: Path) -> None:
    rows = _load_zeroshot_rows(zdir)
    fig, ax = plt.subplots(figsize=(4.4, 2.25))
    y = np.arange(len(rows))[::-1]
    for yi, (_, vals) in zip(y, rows):
        left = 0.0
        for k, v in enumerate(vals):
            ax.barh(yi, v, left=left, height=0.62, color=COLORS[k], edgecolor="white", linewidth=1.0)
            if v >= 0.07:
                ax.text(
                    left + v / 2,
                    yi,
                    f"{round(v * 100)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK if k == 1 else "white",
                )
            left += v
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS],
        CATS,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.02),
        frameon=False,
        fontsize=6.8,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_outcomes.svg")
    plt.close(fig)


def fig_mercator(out_dir: Path) -> None:
    lat = np.linspace(0, 70, 701)
    err = (1 / np.cos(np.radians(lat)) - 1) * 100
    fig, ax = plt.subplots(figsize=(3.2, 2.25))
    for lo, hi, name in [(0, 7, "low"), (25, 45, "zone_edge"), (52, 64, "high")]:
        ax.axvspan(lo, hi, color="#f0efec", lw=0)
        ax.text((lo + hi) / 2, 230, name, ha="center", va="top", fontsize=6.8, color=INK2)
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(69, 1.25, "1% tolerance", ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.plot(lat, err, color=LINE, lw=1.6)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0.01, 300)
    ax.set_xlim(0, 70)
    ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100%"])
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70], ["0°", "10", "20", "30", "40", "50", "60", "70°"])
    ax.set_xlabel("Latitude of the data")
    ax.set_ylabel("Length error in EPSG:3857")
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / "fig_mercator.svg")
    plt.close(fig)


def fig_rsi_gates(out_dir: Path, eval_dir: Path) -> bool:
    seed_p = eval_dir / "seed_e4b_nothink.json"
    if not seed_p.exists():
        return False
    seed_ev = json.loads(seed_p.read_text())

    methods = [
        ("Gen-0 Seed", None),
        ("naive", "naive"),
        ("preserve\n(DarwinX)", "preserve"),
        ("metamorphic\n(Ours)", "meta"),
    ]
    metrics = [
        ("ALL", "correct", "Test ALL (n=150)", "#2a78d6"),
        ("HELDOUT_3", "correct", "Held-out (n=60)", "#008300"),
        ("ALL", "lucky", "Lucky hacks (↓)", "#eda100"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5), sharey=True)
    x = np.arange(len(methods))
    rng = np.random.default_rng(0)

    for ax, (section, key, title, color) in zip(axes, metrics):
        means, stds, raw_pts = [], [], []
        for _, g in methods:
            if g is None:
                vals = [seed_ev[section][key] * 100]
            else:
                vals = []
                for s in range(4):
                    p = eval_dir / f"v3_gis_s{s}_{g}.json"
                    if p.exists():
                        vals.append(json.loads(p.read_text())[section][key] * 100)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            raw_pts.append(vals)

        bars = ax.bar(x, means, width=0.56, color=color, alpha=0.82, edgecolor="white", linewidth=0.8)
        ax.errorbar(
            x[1:],
            means[1:],
            yerr=stds[1:],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
        for xi, vals in zip(x, raw_pts):
            if len(vals) > 1:
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(xi + jitter, vals, color=INK, s=14, zorder=5, alpha=0.85)
        for xi, m in zip(x, means):
            ax.text(xi, min(m + (stds[xi] if xi > 0 else 0) + 3.0, 102), f"{m:.1f}%", ha="center", va="bottom", fontsize=6.8)

        ax.set_title(title, fontsize=8.5, pad=6)
        ax.set_xticks(x, [m[0] for m in methods], fontsize=7.2)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "fig_rsi_gates.svg")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SVG figures for meta-rsi")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--zeroshot-dir", default="results/zero_shot_v1")
    ap.add_argument("--eval-dir", default="runs/eval150")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    fig_outcomes(out_dir, Path(a.zeroshot_dir))
    fig_mercator(out_dir)
    ok_rsi = fig_rsi_gates(out_dir, Path(a.eval_dir))
    print(f"Wrote figures to {out_dir} (fig_rsi_gates={'ok' if ok_rsi else 'skipped'})")


if __name__ == "__main__":
    main()
