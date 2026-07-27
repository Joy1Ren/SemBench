#!/usr/bin/env python3
"""
Plot SemBench plan-evaluation results.
Edit the CONFIGURATION block below, then run:  python plot_results.py
"""

import json
import os
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
USE_CASE     = "ecomm"
SCALE_FACTOR = 2000
# id_list = [1,2,3,4,5,6,7,9,11,12,13] #ecomm 500
id_list = [1,3,4,5,6,7,8,12,13]
QUERIES = [f"Q{id}" for id in id_list]
RUNNERS      = ["palimpzest", "execute_oracle_sampler_agent"]
METRICS_DIR  = f"files/{USE_CASE}/metrics/sf_{SCALE_FACTOR}"
SAMPLING_DIR = f"agent_cost_model/sampling/{USE_CASE}/sf_{SCALE_FACTOR}"
OUTPUT_DIR   = f"results/{USE_CASE}/sf_{SCALE_FACTOR}"   # directory where PNGs are saved
# ─────────────────────────────────────────────────────────────────────────────

# Only FLASH_LITE and FLASH are plotted from palimpzest; others are ignored.
MODEL_LABELS = {
    "GOOGLE_GEMINI_2_5_FLASH_LITE": "low",
    "GOOGLE_GEMINI_2_5_FLASH":      "medium",
}

# Pretty display names for runners (keyed by the filename stem without .json).
# Extend as needed; unrecognized runners fall back to their raw name.
RUNNER_DISPLAY_NAMES = {
    "palimpzest":                       "Palimpzest",
    "execute_oracle_sampler_agent":     "Agent",
    "execute_oracle_agent":             "Execute+Oracle Agent",
    "customCost_oracle_agent":          "CustomCost+Oracle Agent",
    "customCost_oracle_helper_agent":   "CustomCost+Oracle+Helper Agent",
}

# Slot 0 is shared by both palimpzest tiers (differentiated by marker shape).
# Agent runners are assigned slots 1, 2, 3, ... in order.
PALETTE = [
    "#1f77b4",  # blue   -> palimpzest (low and medium share this color)
    "#ff7f0e",  # orange -> 1st agent runner
    "#2ca02c",  # green  -> 2nd agent runner
    "#d62728",  # red    -> 3rd agent runner
    "#9467bd",  # purple -> 4th agent runner
    "#8c564b",  # brown
    "#e377c2",  # pink
]

# Markers that distinguish palimpzest model tiers within their shared color.
PZ_MARKERS = {
    "low":    "o",  # circle
    "medium": "s",  # square
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_palimpzest(path):
    """
    Parse palimpzest.json (keys: {qid}_{MODEL}_{runcount}).
    Returns {qid_str: {model_name: {cost, quality, latency}}} averaged over runcounts.
    Only includes models present in MODEL_LABELS.
    """
    with open(path) as f:
        raw = json.load(f)

    groups = defaultdict(list)
    for key, entry in raw.items():
        if entry.get("scale_factor") != SCALE_FACTOR:
            continue
        parts = key.split("_")
        # Old-style keys like "1_1" have no model embedded; skip them.
        if len(parts) < 3:
            continue
        try:
            qid = int(parts[0])
            int(parts[-1])  # run count must be an integer
        except ValueError:
            continue
        model_name = "_".join(parts[1:-1])
        if model_name not in MODEL_LABELS:
            continue
        groups[(qid, model_name)].append(entry)

    result = defaultdict(dict)
    for (qid, model), entries in groups.items():
        def avg(field):
            vals = [e[field] for e in entries if e.get(field) is not None]
            return sum(vals) / len(vals) if vals else None
        result[f"Q{qid}"][model] = {
            "cost":    avg("cost"),
            "quality": avg("quality"),
            "latency": avg("latency"),
        }
    return dict(result)


def _avg_runs(entry):
    """
    Average run1/run2 sub-dicts if present; fall back to top-level cost/quality/latency.
    """
    if "run1" in entry and "run2" in entry:
        r1, r2 = entry["run1"], entry["run2"]
        cost = ((r1.get("cost") or 0) + (r2.get("cost") or 0)) / 2
        q1, q2 = r1.get("quality"), r2.get("quality")
        qual = (q1 + q2) / 2 if (q1 is not None and q2 is not None) else (q1 if q1 is not None else q2)
        lat  = ((r1.get("latency") or 0) + (r2.get("latency") or 0)) / 2
        return cost, qual, lat
    return entry.get("cost"), entry.get("quality"), entry.get("latency")


def load_agent(path):
    """
    Parse an _agent json (keys: {qid}_{opt_run}).
    Returns a list of per-(query, opt_run) dicts -- all opt runs kept as separate points.
    run1/run2 within each opt run are averaged.
    """
    with open(path) as f:
        raw = json.load(f)

    points = []
    for key, entry in raw.items():
        if entry.get("scale_factor") != SCALE_FACTOR:
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        try:
            qid, opt_run = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        cost, qual, lat = _avg_runs(entry)
        points.append({
            "query_id":              f"Q{qid}",
            "opt_run":               opt_run,
            "exec_cost":             cost if cost is not None else 0.0,
            "quality":               qual,
            "latency":               lat,
            "agent_cost":            entry.get("agent_cost") or 0.0,
            "subset_execution_cost": entry.get("subset_execution_cost") or 0.0,
            "oracle_cost":           entry.get("oracle_cost") or 0.0,
            "plans_executed":        entry.get("plans_executed"),
            "plans_written":         entry.get("plans_written"),
        })
    return points


def load_all():
    data = {}
    for runner in RUNNERS:
        path = os.path.join(METRICS_DIR, f"{runner}.json")
        if runner == "palimpzest":
            data[runner] = ("palimpzest", load_palimpzest(path))
        else:
            data[runner] = ("agent", load_agent(path))
    return data


def load_sampling_costs():
    """Return {qid_str: target_cost} from sampling_results.json, e.g. {"Q1": 0.004775}."""
    path = os.path.join(SAMPLING_DIR, "sampling_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {f"Q{v['query_id']}": v["target_cost"] for v in raw.values() if "target_cost" in v}


# ── Color assignment ──────────────────────────────────────────────────────────

def assign_colors(all_data):
    """
    Both palimpzest tiers share PALETTE[0].
    Agent runners are assigned PALETTE[1], [2], ... in order.
    """
    colors = {
        ("palimpzest", "low"):    PALETTE[0],
        ("palimpzest", "medium"): PALETTE[0],
    }
    ci = 1
    for runner, (rtype, _) in all_data.items():
        if rtype == "agent":
            colors[("agent", runner)] = PALETTE[ci % len(PALETTE)]
            ci += 1
    return colors


# ── Scatter figures (Fig 1 & 2) ───────────────────────────────────────────────

def make_scatter(all_data, colors, with_opt_cost, sampling_costs=None):
    n_q   = len(QUERIES)
    ncols = min(n_q, 4)
    nrows = (n_q + ncols - 1) // ncols

    title = (
        f"{USE_CASE} (scale factor: {SCALE_FACTOR}) -- "
        + ("with opt cost" if with_opt_cost else "without opt cost")
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)
    ax_flat = axes.flatten()

    for qi, qid in enumerate(QUERIES):
        ax = ax_flat[qi]
        ax.set_title(qid, fontsize=11)
        ax.set_xlabel("Cost ($)", fontsize=9)
        ax.set_ylabel("Quality", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, alpha=0.3)

        for runner, (rtype, rdata) in all_data.items():
            if rtype == "palimpzest":
                for model, vals in rdata.get(qid, {}).items():
                    label  = MODEL_LABELS.get(model)
                    color  = colors.get(("palimpzest", label), "#aaaaaa")
                    marker = PZ_MARKERS.get(label, "o")
                    x, y = vals["cost"], vals["quality"]
                    if x is None or y is None:
                        continue
                    ax.scatter(x, y, marker=marker, s=100, color=color, zorder=3)

            else:  # agent -- every opt_run is a separate star
                color = colors.get(("agent", runner), "#aaaaaa")
                tc = (sampling_costs or {}).get(qid, 0.0)
                for pt in rdata:
                    if pt["query_id"] != qid:
                        continue
                    x = pt["exec_cost"]
                    if with_opt_cost:
                        x += pt["agent_cost"] + pt["subset_execution_cost"] + pt["oracle_cost"] + tc
                    y = pt["quality"]
                    if y is None:
                        continue
                    ax.scatter(x, y, marker="*", s=160, color=color, zorder=3)

    # Hide unused axes
    for i in range(n_q, len(ax_flat)):
        ax_flat[i].set_visible(False)

    # Global legend
    handles = []
    pz_color = colors[("palimpzest", "low")]
    pz_seen  = False
    for runner, (rtype, _) in all_data.items():
        if rtype == "palimpzest" and not pz_seen:
            for label in ["low", "medium"]:
                marker = PZ_MARKERS[label]
                handles.append(mlines.Line2D(
                    [], [], color=pz_color, marker=marker, linestyle="None",
                    markersize=9, label=label,
                ))
            pz_seen = True
        elif rtype == "agent":
            color   = colors[("agent", runner)]
            display = RUNNER_DISPLAY_NAMES.get(runner, runner)
            handles.append(mlines.Line2D(
                [], [], color=color, marker="*", linestyle="None",
                markersize=11, label=display,
            ))

    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0), fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    return fig


# ── Table figures (Fig 3+) ────────────────────────────────────────────────────

def _fmt_cost(c):
    if c is None:
        return "N/A"
    if c == 0.0:
        return "$0.00"
    if c < 0.001:
        return f"${c*1000:.3f}e-3"
    return f"${c:.4f}"


def _fmt_q(q):
    return "N/A" if q is None else f"{q:.3f}"


def _fmt_n(n):
    if n is None:
        return "N/A"
    return f"{n:.1f}" if isinstance(n, float) else str(n)


def _render_table(rows, col_headers, title):
    """Render a list of row-lists into a matplotlib table figure."""
    n_rows = len(rows)
    n_cols = len(col_headers)
    fig_h  = max(3.0, 0.45 * (n_rows + 2))
    fig_w  = max(14, 1.8 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    t = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(col=list(range(n_cols)))

    for j in range(n_cols):
        cell = t[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.08)

    for i in range(n_rows):
        fc = "#eaf2fb" if i % 2 == 0 else "white"
        for j in range(n_cols):
            t[(i + 1, j)].set_facecolor(fc)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_table_figures(all_data, sampling_costs=None):
    """
    Returns a list of (fig, filename_stem) tuples -- one table per runner.
    Palimpzest produces two tables (low / medium); each agent runner gets one.
    """
    col_headers = [
        "Query", "Opt Run",
        "Exec Cost", "Quality", "# Plans",
        "Opt Cost", "Agent Cost", "Subset Exec Cost", "Oracle Cost", "Sampling Cost",
    ]
    figs = []

    for runner, (rtype, rdata) in all_data.items():
        display = RUNNER_DISPLAY_NAMES.get(runner, runner)

        if rtype == "palimpzest":
            for model_key in ["GOOGLE_GEMINI_2_5_FLASH_LITE", "GOOGLE_GEMINI_2_5_FLASH"]:
                label = MODEL_LABELS[model_key]
                rows  = []
                for qid in QUERIES:
                    vals = rdata.get(qid, {}).get(model_key)
                    if vals is None:
                        continue
                    rows.append([
                        qid, "--",
                        _fmt_cost(vals["cost"]),
                        _fmt_q(vals["quality"]),
                        "N/A",
                        "$0.00", "N/A", "N/A", "N/A", "N/A",
                    ])
                if not rows:
                    continue
                title = f"{display} ({label}) -- {USE_CASE} sf={SCALE_FACTOR}"
                fig = _render_table(rows, col_headers, title)
                figs.append((fig, f"table_pz_{label}"))

        else:  # agent runner
            rows = []
            for qid in QUERIES:
                pts = sorted(
                    [p for p in rdata if p["query_id"] == qid],
                    key=lambda p: p["opt_run"],
                )
                for pt in pts:
                    sc  = (sampling_costs or {}).get(qid, 0.0)
                    opt = pt["agent_cost"] + pt["subset_execution_cost"] + pt["oracle_cost"] + sc
                    rows.append([
                        qid, str(pt["opt_run"]),
                        _fmt_cost(pt["exec_cost"]),
                        _fmt_q(pt["quality"]),
                        _fmt_n(pt["plans_executed"]),
                        _fmt_cost(opt),
                        _fmt_cost(pt["agent_cost"]),
                        _fmt_cost(pt["subset_execution_cost"]),
                        _fmt_cost(pt["oracle_cost"]),
                        _fmt_cost(sc) if sc else "N/A",
                    ])
            if not rows:
                continue
            title = f"{display} -- {USE_CASE} sf={SCALE_FACTOR}"
            fig = _render_table(rows, col_headers, title)
            figs.append((fig, f"table_{runner}"))

    return figs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data   = load_all()
    colors     = assign_colors(all_data)

    sampling_costs = load_sampling_costs()

    fig1 = make_scatter(all_data, colors, with_opt_cost=False, sampling_costs=sampling_costs)
    fig2 = make_scatter(all_data, colors, with_opt_cost=True,  sampling_costs=sampling_costs)
    table_figs = make_table_figures(all_data, sampling_costs=sampling_costs)

    out1 = os.path.join(OUTPUT_DIR, "costQuality_without_opt_cost.png")
    out2 = os.path.join(OUTPUT_DIR, "costQuality_with_opt_cost.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")

    saved = [out1, out2]
    for fig, stem in table_figs:
        path = os.path.join(OUTPUT_DIR, f"{stem}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)

    print("Saved:\n" + "\n".join(f"  {p}" for p in saved))
    plt.show()


if __name__ == "__main__":
    main()
