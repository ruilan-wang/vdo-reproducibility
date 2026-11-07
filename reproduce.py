"""Commands for reproducing VDO experiments, numerical figures and paper assets."""

import os

for variable in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import unittest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["figures", "export-paper"]:
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, default=ROOT / "outputs" / name)
        p.add_argument("--data", type=Path, default=ROOT / "data/supplement")
        p.add_argument("--psro-data", type=Path, default=ROOT / "data/psro")
    p = sub.add_parser("paper-assets")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/paper-assets")
    sub.add_parser("verify-data")
    sub.add_parser("smoke")
    p = sub.add_parser("matrix")
    p.add_argument("--size", type=int, default=25)
    p.add_argument("--mode", default="vdo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--events", type=int, default=200)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/runs/supplement")
    p = sub.add_parser("supplement")
    p.add_argument("--family", choices=["all", "matrix", "efg"], default="all")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/runs/supplement")
    p = sub.add_parser("psro")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--events", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--methods", nargs="+")
    p = sub.add_parser("plot-psro")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument(
        "--game",
        choices=["kuhn", "leduc", "goofspiel4", "blotto", "poker_4r2s", "liars_dice_5"],
        required=True,
    )
    p.add_argument("--selected-beta", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-data":
        manifest = json.loads((ROOT / "data/checksums.json").read_text())
        for name, expected in manifest.items():
            path = ROOT / name
            if (
                not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise SystemExit(f"Missing or altered input: {name}")
        print(f"Verified {len(manifest)} input files.")
    elif args.command == "smoke":
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    elif args.command in ["figures", "export-paper"]:
        from vdo import figures, table
        from vdo.psro_figures import generate

        figures.generate(args.output, args.data, args.psro_data)
        table.generate(args.output, args.psro_data)
        generate(args.output, args.psro_data)
        entries = json.loads((ROOT / "data/figures.json").read_text())
        for entry in entries:
            entry["input_root"] = str(
                args.psro_data
                if entry["kind"] == "psro"
                or (entry["figure"] == 8 and entry["panel"] == "b")
                else args.data
            )
        (args.output / "figure_sources.json").write_text(
            json.dumps(entries, indent=2) + "\n"
        )
        print(
            "Generated all 36 panels in Figures 1–13 from numerical records, and Table 2."
        )
    elif args.command == "paper-assets":
        args.output.mkdir(parents=True, exist_ok=True)
        entries = json.loads((ROOT / "data/figures.json").read_text())
        for item in entries:
            src = ROOT / item["file"]
            shutil.copyfile(src, args.output / src.name)
        print("Copied the 36 supplied manuscript assets without redrawing them.")
    elif args.command == "plot-psro":
        from vdo.psro_figures import plot

        plot(args.results, args.output, args.game, args.selected_beta)
    elif args.command == "matrix":
        from vdo.runs import supplement_job

        print(
            supplement_job(
                (
                    "matrix",
                    (args.size, args.mode, args.seed, args.events),
                    str(args.output),
                )
            )
        )
    elif args.command == "supplement":
        from vdo.runs import supplement_tasks, execute, supplement_job

        jobs = [
            (family, spec, str(args.output))
            for family, spec in supplement_tasks(args.family)
        ]
        execute(jobs, supplement_job, args.workers)
    elif args.command == "psro":
        from vdo.runs import psro

        psro(
            args.config,
            args.output,
            args.workers,
            args.events,
            args.seeds,
            args.methods,
        )


if __name__ == "__main__":
    main()
