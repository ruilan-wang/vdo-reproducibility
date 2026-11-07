"""Seeded experiment execution with atomic, configuration-checked checkpoints."""

from concurrent.futures import ProcessPoolExecutor
import contextlib
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from vdo import experiment, matrix

ROOT = Path(__file__).resolve().parents[1]
SCALES = {
    "kuhn": 2.0,
    "leduc": 13.0,
    "goofspiel4": 1.0,
    "blotto": 1.0,
    "poker_4r2s": 13.0,
    "liars_dice_5": 1.0,
}


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def psro_result_path(root, oracle, game):
    """Locate either a supplied aggregate or a fresh runner's aggregate."""
    root = Path(root)
    supplied = root / oracle / (game + ".json")
    fresh = root / oracle / game / "all_results.json"
    candidates = [p for p in (supplied, fresh) if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one PSRO aggregate for {oracle}/{game} under {root}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(temporary, "wt") as stream:
            json.dump(value, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ["numpy", "scipy", "open-spiel", "cvxopt"]},
    }


def source_hash():
    digest = hashlib.sha256()
    for p in sorted([*(ROOT / "vdo").glob("*.py"), *(ROOT / "psro_v2").glob("*.py")]):
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def cached_run(path, settings, function):
    identity = {
        "settings": settings,
        "source_sha256": source_hash(),
        "runtime": runtime(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        result = load_json(path)
        if result.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{path}: settings, source, or environment differ; use a fresh output directory."
            )
        return result
    start = time.perf_counter()
    result = function()
    result.update(
        fingerprint=fingerprint,
        provenance=identity,
        wall_seconds=time.perf_counter() - start,
    )
    write_json(path, result)
    return result


def efg_filename(spec):
    game, mode, seed, eta, events = spec
    return f"{game}_{mode}_eta{eta:.8g}_seed{seed}_q{events}.json.gz"


def run_efg_spec(spec):
    game, mode, seed, eta, events = spec
    cfg = load_json(ROOT / f"configs/psro/exact/{game}.json")["global"]
    cfg.update(max_br=events, rne=True)
    beta = (
        "inf"
        if mode == "odo"
        else 1 if mode == "vdo" else int(mode.removeprefix("dvdo_b"))
    )
    method = {
        "name": mode,
        "meta_strategy_method": "vdo",
        "vdo_beta": beta,
        "vdo_eta": eta,
        "vdo_window_scale": 50,
        "normalize_payoffs": True,
    }
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        rows = experiment.run_single(cfg, method, seed, endpoint_only=True)
    scale = SCALES[game]
    points = [
        {
            "br_count": row["br_count"],
            "online": row["exploitability"] / (2 * scale),
            "rne": row["rne_exploitability"] / (2 * scale),
        }
        for row in rows
    ]
    return {
        "spec": list(spec),
        "payoff_scale": scale,
        "training_payoffs": "normalised",
        "metric": "normalised half NashConv",
        "points": points,
    }


def supplement_tasks(family="all"):
    tasks = []
    if family in ["matrix", "all"]:
        tasks.extend(
            ("matrix", (n, mode, seed, 200))
            for n in [25, 50, 100]
            for mode in matrix.BETAS
            for seed in range(1, 11)
        )
        tasks.extend(
            ("matrix", (50, mode, seed, 3000))
            for mode in [
                "odo",
                "vdo",
                "dvdo_b50",
                "dvdo_b20",
                "dvdo_b10",
                "fixed_lambda10",
            ]
            for seed in range(1, 11)
        )
    if family in ["efg", "all"]:
        modes = ["odo", "vdo", "dvdo_b2", "dvdo_b5", "dvdo_b10", "dvdo_b20", "dvdo_b50"]
        tasks.extend(
            ("efg", ("kuhn", mode, seed, eta, 200))
            for eta in [0.01, 0.05, 1 / 11, 0.2, 0.5]
            for mode in modes
            for seed in range(1, 11)
        )
        tasks.extend(
            ("efg", ("leduc", mode, seed, 1 / 11, 200))
            for mode in modes
            for seed in range(1, 11)
        )
    return tasks


def supplement_job(job):
    family, spec, output = job
    path = Path(output) / (
        matrix.filename(spec) if family == "matrix" else efg_filename(spec)
    )
    run = matrix.run_spec if family == "matrix" else run_efg_spec
    result = cached_run(path, {"family": family, "spec": spec}, lambda: run(spec))
    return {"file": str(path), "last": result["points"][-1]}


def execute(jobs, function, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            print(
                json.dumps({"done": i, "total": len(jobs), **function(job)}), flush=True
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(function, jobs), 1):
                print(json.dumps({"done": i, "total": len(jobs), **result}), flush=True)


def psro_job(job):
    cfg, method, seed, output = job
    path = Path(output) / "checkpoints" / f"{method['name']}_seed_{seed}.json"
    settings = {"global": cfg, "method": method, "seed": seed}

    def calculate():
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            points = experiment.run_single(cfg, method, seed)
        return {"points": points, "metric": "native full NashConv"}

    result = cached_run(path, settings, calculate)
    return {"file": str(path), "last": result["points"][-1]}


def psro(config, output, workers=1, events=None, seeds=None, methods=None):
    data = load_json(config)
    cfg = data["global"]
    if cfg.get("n_players") != 2 or cfg.get("oracle_type") != "BR":
        raise ValueError("The paper runner supports two-player BR experiments.")
    if (
        cfg.get("exact_payoffs") is not True
        or cfg.get("per_player_windows") is not True
    ):
        raise ValueError("This runner requires exact payoffs and per-player windows.")
    if events is not None:
        cfg["max_br"] = events
    if cfg["max_br"] < 1:
        raise ValueError("events must be positive")
    seeds = seeds or list(
        range(cfg.get("seed", 1), cfg.get("seed", 1) + cfg["num_runs"])
    )
    experiments = data["experiments"]
    if methods:
        unknown = set(methods) - {x["name"] for x in experiments}
        if unknown:
            raise ValueError(f"Methods not in configuration: {sorted(unknown)}")
        experiments = [x for x in experiments if x["name"] in methods]
    execute(
        [(cfg, m, s, str(output)) for m in experiments for s in seeds],
        psro_job,
        workers,
    )
    aggregate = {"game_name": cfg.get("game_label", cfg["game_name"])}
    for method in experiments:
        runs = [
            load_json(Path(output) / "checkpoints" / f"{method['name']}_seed_{s}.json")[
                "points"
            ]
            for s in seeds
        ]
        if any(len(r) != cfg["max_br"] for r in runs):
            raise ValueError("A checkpoint does not contain every requested event.")
        rows = []
        for index in range(cfg["max_br"]):
            row = {"br_count": index + 1}
            for source, name in [
                ("elapsed_time", "time"),
                ("exploitability", "exploitability"),
                ("rne_exploitability", "rne_exploitability"),
            ]:
                values = [r[index][source] for r in runs]
                row["avg_" + name] = float(np.mean(values))
                row["std_" + name] = float(np.std(values, ddof=0))
            rows.append(row)
        aggregate[method["name"]] = {
            "meta_strategy_method": method["name"],
            "num_runs": len(runs),
            "results_by_br": rows,
        }
    write_json(Path(output) / "all_results.json", aggregate)
    write_json(
        Path(output) / "configuration.json",
        {"global": cfg, "experiments": experiments, "seeds": seeds},
    )
    return aggregate
