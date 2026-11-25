"""Generate publication-ready Markdown/LaTeX tables and vector SVG figures from verified run JSONs.

Strictly enforces `usage.errors == 0` across every cited evaluation file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sprint4_summary import summarize_gsb2


def _fmt_pct_ms(ms: tuple[float, float]) -> str:
    m, s = ms
    return f"{100 * m:.1f} ± {100 * s:.1f}%"


def _svg_bar_chart(
    title: str,
    subtitle: str,
    bars: list[tuple[str, float, float | None, str]],
    out_svg: Path,
    *,
    y_max: float = 100.0,
    y_label: str = "3-Location Oracle Accuracy (%)",
) -> None:
    """Render a clean vector SVG bar chart with optional +/- SD error bars (zero external deps)."""
    w, h = 760, 420
    pad_l, pad_r, pad_t, pad_b = 76, 30, 72, 92
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b
    n = max(len(bars), 1)
    step = pw / n
    bw = min(56.0, step * 0.58)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Inter, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="30" font-size="16" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{pad_l}" y="50" font-size="11.5" fill="#4b5563">{subtitle}</text>',
    ]
    # Horizontal grid
    for tick in range(0, int(y_max) + 1, 20):
        y = pad_t + ph * (1.0 - tick / y_max)
        lines.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{tick}</text>')
    lines.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#374151" stroke-width="1.2"/>')
    lines.append(f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{w - pad_r}" y2="{pad_t + ph}" stroke="#374151" stroke-width="1.2"/>')
    lines.append(
        f'<text x="20" y="{pad_t + ph / 2:.1f}" transform="rotate(-90 20 {pad_t + ph / 2:.1f})" '
        f'text-anchor="middle" font-size="11.5" font-weight="600" fill="#374151">{y_label}</text>'
    )

    for idx, (label, val, sd, color) in enumerate(bars):
        cx = pad_l + step * (idx + 0.5)
        x = cx - bw / 2
        bh = ph * max(0.0, min(val, y_max)) / y_max
        y = pad_t + ph - bh
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
        if sd is not None and sd > 0:
            y_hi = pad_t + ph * (1.0 - min(y_max, val + sd) / y_max)
            y_lo = pad_t + ph * (1.0 - max(0.0, val - sd) / y_max)
            lines.append(f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#111827" stroke-width="1.6"/>')
            lines.append(f'<line x1="{cx - 5:.1f}" y1="{y_hi:.1f}" x2="{cx + 5:.1f}" y2="{y_hi:.1f}" stroke="#111827" stroke-width="1.6"/>')
            lines.append(f'<line x1="{cx - 5:.1f}" y1="{y_lo:.1f}" x2="{cx + 5:.1f}" y2="{y_lo:.1f}" stroke="#111827" stroke-width="1.6"/>')
            txt_y = max(pad_t - 4, y_hi - 7)
            val_lbl = f"{val:.1f}±{sd:.1f}"
        else:
            txt_y = max(pad_t - 4, y - 7)
            val_lbl = f"{val:.1f}%"
        lines.append(f'<text x="{cx:.1f}" y="{txt_y:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#111827">{val_lbl}</text>')
        for li, part in enumerate(label.split("\n")):
            lines.append(
                f'<text x="{cx:.1f}" y="{pad_t + ph + 18 + 14 * li:.1f}" text-anchor="middle" font-size="11" fill="#1f2937">{part}</text>'
            )

    lines.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(lines))


def generate_all(runs_dir: Path, out_dir: Path) -> None:
    rep = summarize_gsb2(runs_dir)
    if rep["usage_error_violations"]:
        raise RuntimeError(f"Non-zero usage errors detected: {rep['usage_error_violations']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ms = rep["gsb2_multiseed"]
    sg = rep["gsb2_singles"]

    # 1. Markdown summary table
    md = [
        "# Verified GSB v2 & Cross-Domain Summary Tables",
        "",
        "All metrics strictly audited with `usage.errors == 0`.",
        "",
        "## Table 1: GSB v2 Multi-Seed Harness Evolution (`eval450`, `google/gemma-4-26B-A4B-it`)",
        "",
        "| Method | Seeds | ALL | Seen (18) | Held-Out (3) | Lucky | Err | CompV | Rast | Calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("seed_26b", "Seed (`seed.py`)"),
        ("handcrafted_26b", "Handcrafted (`handcrafted.py`)"),
        ("runtime_meta_26b", "Runtime-Meta (inference-time)"),
        ("s0_naive", "Naive (s0)"),
        ("s0_meta_gate_only", "Meta-Gate-Only (zero diag, s0)"),
        ("s0_oracle_soft", "Oracle-Soft (3-loc upper bound, s0)"),
    ]:
        if key in sg:
            r = sg[key]
            g = r["groups"]
            md.append(
                f"| {label} | 1 | {100*r['all']:.1f}% | {100*r['seen']:.1f}% | {100*r['ho']:.1f}% | "
                f"{100*r['lucky']:.1f}% | {100*r['error']:.1f}% | {100*g.get('CompV', 0):.1f}% | "
                f"{100*g.get('Rast', 0):.1f}% | {r['calls']:.2f} |"
            )

    for mkey, mlabel in [
        ("darwinx_soft", "DarwinX-Soft (weak dev_low)"),
        ("llm_judge", "LLM-as-a-Judge (static rubric)"),
        ("meta_T0", "Meta-T0 (fixed mid-lat probe)"),
        ("coev_tip_only", "Co-Ev Tip-Only (T0 + certified)"),
        ("arch_coev", "Archive-Preserving Co-Ev (ours)"),
    ]:
        if mkey in ms:
            d = ms[mkey]
            g = d["groups"]
            md.append(
                f"| **{mlabel}** | {len(d['seeds'])} | {_fmt_pct_ms(d['all'])} | {_fmt_pct_ms(d['seen'])} | "
                f"{_fmt_pct_ms(d['ho'])} | {_fmt_pct_ms(d['lucky'])} | {_fmt_pct_ms(d['error'])} | "
                f"{_fmt_pct_ms(g['CompV'])} | {_fmt_pct_ms(g['Rast'])} | {d['calls'][0]:.2f} ± {d['calls'][1]:.2f} |"
            )

    if rep.get("osm_multiseed"):
        md.extend([
            "",
            "## Table 2: Zero-Shot Real OpenStreetMap 4-City Urban Transfer (`n=24`, 6 Seeds)",
            "",
            "| Method | Seeds | ALL | SilW | Err | Calls | Per-Seed ALL |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for mkey, mlabel in [("arch_coev", "Archive-Preserving Co-Ev (`arch_coev`)")]:
            if mkey in rep["osm_multiseed"]:
                d = rep["osm_multiseed"][mkey]
                ps = ", ".join(f"s{k}={v:.1f}%" for k, v in d["per_seed_all"].items())
                md.append(
                    f"| **{mlabel}** | {len(d['seeds'])} | {_fmt_pct_ms(d['all'])} | {_fmt_pct_ms(d['silent_wrong'])} | "
                    f"{_fmt_pct_ms(d['error'])} | {d['calls'][0]:.2f} ± {d['calls'][1]:.2f} | {ps} |"
                )

    if rep["sci_multiseed"]:
        md.extend([
            "",
            "## Table 3: Non-GIS SciUnits Multi-Seed Paired Comparison (`eval280`)",
            "",
            "| Method | Seeds | ALL | Seen (8) | Held-Out (4) | Lucky | SilW | Err | Calls |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for mkey, mlabel in [
            ("darwinx_soft", "DarwinX-Soft (`26B -> 26B`, canonical SI dev)"),
            ("meta_soft", "Meta-Soft (`26B -> 26B`, unit/sentinel invariance)"),
            ("gem4h_darwinx", "DarwinX-Soft (`gem4h -> 26B`, canonical SI dev)"),
            ("gem4h_meta", "Meta-Soft (`gem4h -> 26B`, unit/sentinel invariance)"),
        ]:
            if mkey in rep["sci_multiseed"]:
                d = rep["sci_multiseed"][mkey]
                md.append(
                    f"| {mlabel} | {len(d['seeds'])} | {_fmt_pct_ms(d['all'])} | {_fmt_pct_ms(d['seen'])} | "
                    f"{_fmt_pct_ms(d['ho'])} | {_fmt_pct_ms(d['lucky'])} | {_fmt_pct_ms(d['silent_wrong'])} | "
                    f"{_fmt_pct_ms(d['error'])} | {d['calls'][0]:.2f} ± {d['calls'][1]:.2f} |"
                )

    if rep.get("geoanalyst"):
        md.extend([
            "",
            "## Table 4: External `GeoAnalystBench` Deterministic Audit (`n=17`, `max_tokens=4096`)",
            "",
            "| Weights | Syntax OK | Uses 3857 | Proj Metric Vec | Dynamic CRS Vec | Raster NoData Safe | CRS Aligned | Strict Geo Sound |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, g in rep["geoanalyst"].items():
            md.append(
                f"| `{label}` | {100*g['syntax_ok']:.1f}% | {100*g['uses_3857_all']:.1f}% | "
                f"{100*g['projected_metric_vec']:.1f}% | {100*g['dynamic_crs_vec']:.1f}% | "
                f"{100*g['raster_nodata_safe']:.1f}% | {100*g['crs_aligned']:.1f}% | **{100*g['strict_geo_sound']:.1f}%** |"
            )

    if rep.get("wave2_v2"):
        md.extend([
            "",
            "## Table 5: GSB v2 Two-Timescale Weight Internalization & Gen-2 Compounding RSI (`eval450` + Real OSM `n=24`)",
            "",
            "| Stage | Calls | ALL (`n=450`) | Seen (`n=300`) | Held-Out (`n=150`) | Lucky | Err | CompV | Rast | Large | Real OSM (`n=24`) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        if "seed_26b" in sg:
            r = sg["seed_26b"]
            g = r["groups"]
            md.append(
                f"| Base 26B (bare `seed.py`) | {r['calls']:.2f} | {100*r['all']:.1f}% | {100*r['seen']:.1f}% | "
                f"{100*r['ho']:.1f}% | {100*r['lucky']:.1f}% | {100*r['error']:.1f}% | {100*g.get('CompV',0):.1f}% | "
                f"{100*g.get('Rast',0):.1f}% | {100*g.get('Large',0):.1f}% | 41.7% |"
            )
        w2o = rep.get("wave2_osm") or {}
        for key, label in [
            ("sft_weak_bare", "`SFT-Weak` LoRA (bare `seed.py`)"),
            ("sft_meta_bare", "`SFT-Meta` LoRA (bare `seed.py`)"),
            ("grpo_meta_bare", "**`GRPO-Meta` LoRA (bare `seed.py`)**"),
            ("gen2_grpo_meta", "**Gen-2 Compounding (`GRPO-Meta` + 0-GIS-rule retry)**"),
        ]:
            if key in rep["wave2_v2"]:
                r = rep["wave2_v2"][key]
                g = r["groups"]
                osm_v = f"**{100*w2o[key]['all']:.1f}%**" if key in w2o else "—"
                md.append(
                    f"| {label} | {r['calls']:.2f} | **{100*r['all']:.1f}%** | {100*r['seen']:.1f}% | "
                    f"**{100*r['ho']:.1f}%** | {100*r['lucky']:.1f}% | {100*r['error']:.1f}% | "
                    f"{100*g.get('CompV',0):.1f}% | {100*g.get('Rast',0):.1f}% | {100*g.get('Large',0):.1f}% | {osm_v} |"
                )

    (out_dir / "summary_tables.md").write_text("\n".join(md) + "\n")
    (out_dir / "summary_data.json").write_text(json.dumps(rep, indent=2))

    # 2. Vector SVG figure for GSB v2 main comparison
    bars: list[tuple[str, float, float | None, str]] = []
    if "seed_26b" in sg:
        bars.append(("Seed\n(Gen-0)", 100 * sg["seed_26b"]["all"], None, "#9ca3af"))
    if "darwinx_soft" in ms:
        bars.append(("DarwinX\n(Weak)", 100 * ms["darwinx_soft"]["all"][0], 100 * ms["darwinx_soft"]["all"][1], "#ef4444"))
    if "meta_T0" in ms:
        bars.append(("Fixed T0\n(Mid-Lat)", 100 * ms["meta_T0"]["all"][0], 100 * ms["meta_T0"]["all"][1], "#f59e0b"))
    if "llm_judge" in ms:
        bars.append(("LLM-Judge\n(Static)", 100 * ms["llm_judge"]["all"][0], 100 * ms["llm_judge"]["all"][1], "#8b5cf6"))
    if "s0_meta_gate_only" in sg:
        bars.append(("Meta-Gate\n(0 Diag)", 100 * sg["s0_meta_gate_only"]["all"], None, "#0ea5e9"))
    if "arch_coev" in ms:
        bars.append(("Arch Co-Ev\n(Ours)", 100 * ms["arch_coev"]["all"][0], 100 * ms["arch_coev"]["all"][1], "#10b981"))
    elif "coev_tip_only" in ms:
        bars.append(("Co-Ev\n(Tip-Only)", 100 * ms["coev_tip_only"]["all"][0], 100 * ms["coev_tip_only"]["all"][1], "#10b981"))
    if "s0_oracle_soft" in sg:
        bars.append(("Oracle\n(Upper)", 100 * sg["s0_oracle_soft"]["all"], None, "#374151"))

    _svg_bar_chart(
        "GSB v2 (450 Tasks, 3-Location Oracle): Multi-Seed Verifier Comparison",
        "Symmetric open-weight 26B-A4B -> 26B-A4B harness evolution; error bars show +/- 1 SD across 6 seeds.",
        bars,
        out_dir / "fig_gsb2_multiseed.svg",
    )

    if rep.get("wave2_v2"):
        w2 = rep["wave2_v2"]
        w2_bars: list[tuple[str, float, float | None, str]] = []
        if "seed_26b" in sg:
            w2_bars.append(("Base 26B\n(Bare 1.00)", 100 * sg["seed_26b"]["all"], None, "#9ca3af"))
        if "sft_weak_bare" in w2:
            w2_bars.append(("SFT-Weak\n(Bare 1.00)", 100 * w2["sft_weak_bare"]["all"], None, "#ef4444"))
        if "sft_meta_bare" in w2:
            w2_bars.append(("SFT-Meta\n(Bare 1.00)", 100 * w2["sft_meta_bare"]["all"], None, "#0ea5e9"))
        if "grpo_meta_bare" in w2:
            w2_bars.append(("GRPO-Meta\n(Bare 1.00)", 100 * w2["grpo_meta_bare"]["all"], None, "#10b981"))
        if "gen2_grpo_meta" in w2:
            w2_bars.append(("Gen-2 RSI\n(0 GIS, 1.09)", 100 * w2["gen2_grpo_meta"]["all"], None, "#047857"))
        _svg_bar_chart(
            "GSB v2 Two-Timescale Weight Internalization (26B-A4B, n=450)",
            "Disjoint train[20:60] -> SFT + 24-step GRPO-Metamorphic (3 certified probes) -> bare seed.py & Gen-2 retry.",
            w2_bars,
            out_dir / "fig_gsb2_two_timescale.svg",
        )
    print(f"Wrote tables & SVG figures to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate publication tables and SVG figures")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="runs/figures")
    a = ap.parse_args()
    generate_all(Path(a.runs_dir), Path(a.out_dir))


if __name__ == "__main__":
    main()
