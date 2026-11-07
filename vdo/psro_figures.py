"""Plot native-payoff PSRO records in normalised half-NashConv units."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vdo.figures import COLORS, LABELS, BEST_COLOR
from vdo.runs import SCALES, psro_result_path

ROOT = Path(__file__).resolve().parents[1]


def series(data, game, selected_beta=None):
    """Return exactly the selected means and confidence intervals in plot units."""
    names = [n for n in data if n not in ["game_name", "do_nash"]]
    if selected_beta is not None:
        names = [
            n
            for n in names
            if n
            in ["vdo_odo", "vdo_beta_1", "vdo_fp_hard_br", f"vdo_beta_{selected_beta}"]
        ]
    result = []
    for name in names:
        method = data[name]
        for metric in ["exploitability", "rne_exploitability"]:
            rows = method["results_by_br"]
            key = metric
            if "avg_" + key not in rows[0] and key == "rne_exploitability":
                key = "mrcp_exploitability"
            divisor = 2 * SCALES[game]
            mean = np.array([r["avg_" + key] for r in rows]) / divisor
            ci = (
                1.96
                * np.array([r["std_" + key] for r in rows])
                / (divisor * np.sqrt(method["num_runs"]))
            )
            result.append(
                {
                    "method": name,
                    "metric": metric,
                    "events": [r["br_count"] for r in rows],
                    "mean": mean.tolist(),
                    "ci95": ci.tolist(),
                    "seeds": method["num_runs"],
                }
            )
    return result


def plot(source, output, game, selected_beta=None):
    data = json.loads(Path(source).read_text())
    traces = series(data, game, selected_beta)
    names = list(dict.fromkeys(t["method"] for t in traces))
    if not names:
        raise ValueError("No PSRO methods to plot.")
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 5.61), sharex=True)
    for ax, metric, label in zip(
        axes,
        ["exploitability", "rne_exploitability"],
        ["Online exploitability", "Restricted-NE exploitability"],
    ):
        for name in names:
            mode = {
                "vdo_odo": "odo",
                "vdo_beta_1": "vdo",
                "vdo_fp_hard_br": "fp_hardbr",
            }.get(name)
            if mode is None:
                mode = "dvdo_b" + name.removeprefix("vdo_beta_")
            trace = next(
                t for t in traces if t["method"] == name and t["metric"] == metric
            )
            mean, ci = np.array(trace["mean"]), np.array(trace["ci95"])
            color = (
                BEST_COLOR
                if selected_beta is not None and mode.startswith("dvdo")
                else COLORS[mode]
            )
            xs = trace["events"]
            ax.plot(xs, mean, color=color, lw=2, label=LABELS[mode])
            ax.fill_between(
                xs,
                mean - ci,
                mean + ci,
                color=color,
                alpha=0.2,
                lw=0,
            )
        ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.2)
    titles = {
        "kuhn": "Kuhn poker",
        "leduc": "Leduc poker",
        "goofspiel4": "Goofspiel-4",
        "blotto": "Blotto",
        "poker_4r2s": "Leduc-8",
        "liars_dice_5": "Liar's Dice-5",
    }
    axes[0].set_title("PSRO " + titles[game], fontsize=14)
    axes[0].legend(
        frameon=False, fontsize=8 if len(names) > 5 else 10, loc="upper right"
    )
    axes[1].set_xlabel("BR events per player", fontsize=13)
    axes[1].set_xlim(left=1)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return traces


def generate(output, data_root=None):
    """Generate the 21 PSRO panels in Figures 1–3, 5 and 11–13."""
    output = Path(output)
    data_root = Path(data_root) if data_root is not None else ROOT / "data/psro"
    specifications = json.loads((ROOT / "data/figures.json").read_text())
    numerical = []
    for spec in specifications:
        if spec["kind"] != "psro":
            continue
        source = psro_result_path(data_root, spec["oracle"], spec["game"])
        stem = f"figure_{spec['figure']:02d}_{spec['panel']}"
        traces = plot(source, output / stem, spec["game"], spec["selected_beta"])
        numerical.append(
            {
                "figure": spec["figure"],
                "panel": spec["panel"],
                "oracle": spec["oracle"],
                "game": spec["game"],
                "source": str(source),
                "traces": traces,
            }
        )
    (output / "psro_plot_values.json").write_text(
        json.dumps(numerical, indent=2) + "\n"
    )
    return numerical
