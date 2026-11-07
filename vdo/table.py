"""Generate Table 2 from native-payoff PSRO endpoint means."""

import csv
from decimal import Decimal
import json
from pathlib import Path
from vdo.runs import load_json, psro_result_path

ROOT = Path(__file__).resolve().parents[1]


def generate(output, psro_data=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    psro_data = Path(psro_data) if psro_data is not None else ROOT / "data/psro"
    selection = json.loads((ROOT / "data/table2_selection.json").read_text())
    titles = {
        "kuhn": "Kuhn poker",
        "leduc": "Leduc poker",
        "goofspiel4": "Goofspiel-4",
        "blotto": "Blotto",
        "poker_4r2s": "Leduc-8",
        "liars_dice_5": "Liar's Dice-5",
    }
    rows = []
    tex = []
    for metric, index in [("online", 2), ("rne", 3)]:
        tex.append("% " + metric)
        for game, title in titles.items():
            for oracle in ["exact", "eps", "sampled"]:
                aggregate = load_json(psro_result_path(psro_data, oracle, game))
                methods = {}
                for name, entry in aggregate.items():
                    if not isinstance(entry, dict):
                        continue
                    last = entry["results_by_br"][-1]
                    if last["br_count"] != 200:
                        raise ValueError(
                            f"{oracle}/{game}/{name}: Table 2 requires event 200."
                        )
                    methods[name] = [
                        entry["num_runs"],
                        last["br_count"],
                        last["avg_exploitability"],
                        last["avg_rne_exploitability"],
                    ]
                beta = selection[oracle][game]
                scale = {"kuhn": 2, "leduc": 13, "poker_4r2s": 13}.get(game, 1)
                names = [
                    "vdo_odo",
                    "vdo_beta_1",
                    f"vdo_beta_{beta}",
                    "vdo_fp_hard_br",
                    "do_nash",
                ]
                values = [
                    (
                        None
                        if k not in methods
                        else Decimal(str(methods[k][index])) / Decimal(2 * scale)
                    )
                    for k in names
                ]

                def fmt(value):
                    if value is None:
                        return "--"
                    if abs(value) < Decimal("1e-9"):
                        return r"$\approx 0$"
                    return format(
                        value.quantize(Decimal(1).scaleb(value.adjusted() - 2)), "f"
                    )

                labels = [fmt(v) for v in values]
                best = fmt(min(values[:3]))
                for i in range(3):
                    if labels[i] == best:
                        labels[i] = (
                            r"$\mathbf{\approx 0}$"
                            if "$" in labels[i]
                            else r"\textbf{" + labels[i] + "}"
                        )
                labels[2] += f" ({beta})"
                tex.append(" & ".join([title, oracle, *labels]) + r" \\")
                for name, value in zip(names, values):
                    if value is not None:
                        rows.append(
                            {
                                "game": game,
                                "oracle": oracle,
                                "metric": metric,
                                "method": name,
                                "events": methods[name][1],
                                "seeds": methods[name][0],
                                "normalised_half_nashconv": str(value),
                                "displayed": fmt(value),
                            }
                        )
    with (output / "table_02.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (output / "table_02_rows.tex").write_text("\n".join(tex) + "\n")
    return rows
