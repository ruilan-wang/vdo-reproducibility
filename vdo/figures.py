"""Generate Figures 4 and 6–10 from the supplied numerical results."""

import json, gzip, math
from pathlib import Path
from functools import lru_cache
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from vdo.runs import psro_result_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/supplement"
PSRO_DATA = ROOT / "data/psro"
OUT = ROOT / "outputs/figures"
BETAS = [1, 2, 5, 10, 20, 50]
FLOOR = 1e-4


BEST_COLOR = "#167A3F"
INK = "#454545"
COLORS = {
    "odo": "#D55E00",
    "vdo": "#0072B2",
    "fp_hardbr": "#7F7F7F",
    "fixed_lambda10": INK,
}
COLORS.update(
    {
        f"dvdo_b{b}": to_hex(plt.cm.YlGn(0.2 + 0.7 * np.log10(b) / 2))
        for b in [2, 5, 10, 20, 50, 100]
    }
)
LABELS = {
    "odo": r"ODO ($\beta=\infty$)",
    "vdo": r"VDO ($\beta=1$)",
    "fp_hardbr": r"FP hard-BR ($\beta\to 0$)",
    "fixed_lambda10": r"Fixed $\lambda=10$",
}
LABELS.update({f"dvdo_b{b}": rf"dVDO ($\beta={b}$)" for b in [2, 5, 10, 20, 50, 100]})
CACHE = {}
SUMMARY = []


def load(game, mode, seed, eta=1 / 11, q=200):
    fn = (
        f"{game}_{mode}_seed{seed}_q{q}.json.gz"
        if game.startswith("matrix")
        else f"{game}_{mode}_eta{eta:.8g}_seed{seed}_q{q}.json.gz"
    )
    if fn not in CACHE:
        with gzip.open(DATA / fn, "rt") as f:
            CACHE[fn] = json.load(f)
    return CACHE[fn]


def count(game):
    return 5 if game == "goof4" else 10


def stats(a):
    a = np.asarray(a)
    return a.mean(0), 1.96 * a.std(0, ddof=0) / np.sqrt(len(a))


def values(game, mode, metric="online", eta=1 / 11, q=200, trajectory=False):
    return np.array(
        [
            (
                [max(0.0, p[metric]) for p in load(game, mode, s, eta, q)["points"]]
                if trajectory
                else max(0.0, load(game, mode, s, eta, q)["points"][-1][metric])
            )
            for s in range(1, count(game) + 1)
        ]
    )


@lru_cache(maxsize=1)
def reported_goofspiel():
    with psro_result_path(PSRO_DATA, "exact", "goofspiel4").open() as stream:
        return json.load(stream)


def endpoint(game, mode, metric="online", eta=1 / 11):
    if game == "goof4" and eta == 1 / 11:
        method = {"odo": "vdo_odo", "vdo": "vdo_beta_1"}.get(mode)
        if method is None:
            method = "vdo_beta_" + mode.removeprefix("dvdo_b")
        entry = reported_goofspiel()[method]
        row = entry["results_by_br"][-1]
        assert row["br_count"] == 200 and entry["num_runs"] == 5
        key = {"online": "exploitability", "rne": "rne_exploitability"}[metric]

        mean = row["avg_" + key] / 2
        ci = 1.96 * (row["std_" + key] / 2) / np.sqrt(entry["num_runs"])
        SUMMARY.append(
            dict(
                game=game,
                mode=mode,
                metric=metric,
                eta=eta,
                mean=mean,
                ci95=float(ci),
                seeds=entry["num_runs"],
                event=200,
                source=str(psro_result_path(PSRO_DATA, "exact", "goofspiel4")),
            )
        )
        return mean, ci
    v = values(game, mode, metric, eta)
    m, c = stats(v)
    SUMMARY.append(
        {
            "game": game,
            "mode": mode,
            "metric": metric,
            "eta": eta,
            "mean": float(m),
            "ci95": float(c),
            "seeds": count(game),
            "per_seed": v.tolist(),
        }
    )
    return m, c


def save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def matrix_plots():
    best = {
        n: min(
            [f"dvdo_b{b}" for b in BETAS[1:]],
            key=lambda m: values(f"matrix{n}", m).mean(),
        )
        for n in [25, 50, 100]
    }
    for n in [25, 50, 100]:
        for baseline in [False, True]:
            fs = (4.6, 5.61)
            sizes = (14, 13, 12, 10 if baseline else 8)
            fig, axes = plt.subplots(2, 1, figsize=fs, sharex=True)
            modes = (
                ["odo", "vdo", best[n], "fp_hardbr"]
                if baseline
                else ["odo", "vdo"] + [f"dvdo_b{b}" for b in BETAS[1:]] + ["fp_hardbr"]
            )
            for ax, metric, ylabel in zip(
                axes,
                ["online", "rne"],
                ["Online exploitability", "Restricted-NE exploitability"],
            ):
                for mode in modes:
                    m, c = stats(values(f"matrix{n}", mode, metric, trajectory=True))
                    x = np.arange(1, len(m) + 1)
                    color = (
                        BEST_COLOR
                        if baseline and mode.startswith("dvdo")
                        else COLORS[mode]
                    )
                    ax.plot(
                        x,
                        np.maximum(m, FLOOR),
                        color=color,
                        ls="--" if mode == "vdo" and not baseline else "-",
                        lw=2,
                        label=LABELS[mode],
                    )
                    ax.fill_between(
                        x,
                        np.maximum(m - c, FLOOR),
                        np.maximum(m + c, FLOOR),
                        color=color,
                        alpha=0.2,
                        lw=0,
                    )
                ax.set_yscale("log")
                ax.set_ylabel(ylabel, fontsize=sizes[1])
                ax.tick_params(labelsize=sizes[2])
                ax.grid(True, alpha=0.2)
                ax.set_xlim(1, 200)
            title = rf"Random ${n}\times{n}$ matrix game"
            axes[1].set_xlabel("BR events per player", fontsize=sizes[1])
            if baseline:
                axes[0].set_title(title, fontsize=sizes[0])
                axes[0].legend(fontsize=sizes[3], frameon=False, loc="upper right")
                fig.tight_layout()
            else:

                fig.suptitle(title, fontsize=sizes[0], y=0.995)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.55, 0.958),
                    ncol=2,
                    fontsize=8,
                    frameon=False,
                    columnspacing=1.1,
                    handlelength=2.4,
                    labelspacing=0.25,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.82))
            save(
                fig, f"figure_{7 if baseline else 6:02d}_{chr(97+[25,50,100].index(n))}"
            )
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for n, marker, line in zip([25, 50, 100], ["o", "s", "^"], ["-", "--", ":"]):
        m, c = zip(
            *[endpoint(f"matrix{n}", "vdo" if b == 1 else f"dvdo_b{b}") for b in BETAS]
        )
        ax.errorbar(
            BETAS,
            m,
            yerr=c,
            marker=marker,
            ls=line,
            lw=2,
            capsize=3,
            color=INK,
            mfc="white" if n == 50 else INK,
            label=f"n={n}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Discount $\beta$", fontsize=13)
    ax.set_xticks(BETAS, [str(b) for b in BETAS])
    ax.set_ylabel("Online exploitability", fontsize=13)
    ax.set_title("Discount sensitivity across matrix sizes", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    save(fig, "figure_09_a")
    return best


def beta_plot(game, title, betas, stem, population=True):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    modes = ["vdo" if b == 1 else f"dvdo_b{b}" for b in betas]

    metrics = [("online", "-", "o", "Online")]
    if population:
        metrics.append(("rne", "--", "s", "RNE"))
    for metric, ls, marker, label in metrics:
        m, c = map(np.array, zip(*[endpoint(game, mode, metric) for mode in modes]))
        v = np.maximum(m, FLOOR)
        ax.errorbar(
            betas,
            v,
            yerr=[v - np.maximum(m - c, FLOOR), np.maximum(m + c, FLOOR) - v],
            color=INK,
            ls=ls,
            lw=1.7,
            capsize=3,
        )
        for beta, mode, y in zip(betas, modes, v):
            col = COLORS[mode]
            ax.plot(
                beta,
                y,
                ls="none",
                marker=marker,
                mec=col,
                mew=1.6,
                mfc=col if metric == "online" else "white",
                ms=6,
                zorder=4,
            )
        ax.plot(
            [],
            [],
            color=INK,
            ls=ls,
            marker=marker,
            mfc=INK if metric == "online" else "white",
            label=label,
        )
    for metric, ls, _, label in metrics:
        m, c = endpoint(game, "odo", metric)
        ax.axhline(
            max(m, FLOOR),
            color=COLORS["odo"],
            ls=ls,
            lw=2,
            label="ODO " + ("online" if metric == "online" else label),
        )
        ax.fill_between(
            [betas[0], betas[-1]],
            max(m - c, FLOOR),
            max(m + c, FLOOR),
            color=COLORS["odo"],
            alpha=0.12,
            lw=0,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(betas, [str(b) for b in betas])
    ax.set_xlabel(r"Discount $\beta$", fontsize=13)
    ax.set_ylabel("Final exploitability", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)
    legend_position = (
        dict(loc="center", bbox_to_anchor=(0.49, 0.72), ncol=2)
        if game == "goof4"
        else dict(loc="best")
    )
    ax.legend(fontsize=8.5, frameon=False, **legend_position)
    fig.tight_layout()
    save(fig, stem)


def eta_plot():
    etas = [0.01, 0.05, 1 / 11, 0.2, 0.5]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for mode in [
        "odo",
        "vdo",
        "dvdo_b2",
        "dvdo_b5",
        "dvdo_b10",
        "dvdo_b20",
        "dvdo_b50",
    ]:
        m, c = zip(*[endpoint("kuhn", mode, eta=e) for e in etas])
        ax.errorbar(
            etas,
            m,
            yerr=c,
            color=COLORS[mode],
            ls="--" if mode == "vdo" else "-",
            marker="o",
            lw=2,
            capsize=3,
            label=LABELS[mode],
        )
    ax.set_xscale("log")
    ax.set_xticks(etas, ["0.01", "0.05", "0.0909", "0.2", "0.5"])
    ax.set_xlabel(r"$\eta$", fontsize=13)
    ax.set_ylabel("Final exploitability", fontsize=13)
    ax.set_title(r"$\eta$ sensitivity on Kuhn", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "figure_04_a")


def long_plot():
    fig, ax = plt.subplots(figsize=(7.0, 3.85))
    rows = []
    for mode in ["odo", "vdo", "dvdo_b10", "dvdo_b20", "dvdo_b50", "fixed_lambda10"]:
        color = COLORS[mode]
        ls = {"vdo": "--", "fixed_lambda10": "-."}.get(mode, "-")
        label = LABELS[mode]
        lw = 2
        v = values("matrix50", mode, q=3000, trajectory=True)
        m, c = stats(v)
        x = np.arange(1, 3001)
        ax.plot(x, m, color=color, ls=ls, lw=lw, label=label)
        ax.fill_between(
            x, np.maximum(m - c, 1e-12), m + c, color=color, alpha=0.2, lw=0
        )
        mm, cc = stats(v[:, -1])
        rows.append({"mode": mode, "mean": float(mm), "ci95": float(cc)})
    ax.set_yscale("log")
    ax.set_xlim(1, 3000)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("BR events per player", fontsize=14)
    ax.set_ylabel("Online exploitability", fontsize=14)
    ax.set_title(r"Random $50\times50$ matrix game", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "figure_10_a")
    return rows


def generate(output=None, data=None, psro_data=None):
    global OUT, DATA, PSRO_DATA
    OUT = Path(output) if output is not None else ROOT / "outputs/figures"
    DATA = Path(data) if data is not None else ROOT / "data/supplement"
    PSRO_DATA = Path(psro_data) if psro_data is not None else ROOT / "data/psro"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.clear()
    reported_goofspiel.cache_clear()
    SUMMARY.clear()
    best = matrix_plots()
    eta_plot()
    specs = [
        ("kuhn", "Kuhn poker", BETAS, "figure_08_a", False),
        ("goof4", "Goofspiel-4", BETAS, "figure_08_b", True),
        ("matrix25", "Random 25x25 matrix game", BETAS, "figure_08_c", True),
        ("matrix50", "Random 50x50 matrix game", BETAS, "figure_08_d", True),
        ("matrix100", "Random 100x100 matrix game", BETAS, "figure_08_e", True),
        ("leduc", "Leduc poker", BETAS, "figure_08_f", True),
    ]
    for args in specs:
        beta_plot(*args)
    long = long_plot()
    report = {"matrix_best": best, "long_horizon": long, "endpoints": SUMMARY}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"matrix_best": best, "long_horizon": long}))
