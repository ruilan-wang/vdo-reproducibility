"""Normalised Gaussian matrix-game experiments."""

import hashlib
import math

import numpy as np
from scipy.optimize import linprog

from psro_v2.meta_strategies import VDOMetaSolver

BETAS = {
    "odo": math.inf,
    "vdo": 1.0,
    "fp_hardbr": 0.0,
    **{f"dvdo_b{b}": b for b in [2, 5, 10, 20, 50]},
}


class MatrixGame:
    """Restricted populations and payoffs exposed to the PSRO meta-solver."""

    def __init__(self, payoffs, seed):
        self.payoffs = payoffs
        rng = np.random.RandomState(seed)
        self.ids = [[int(rng.choice(n))] for n in payoffs.shape]

    def get_policies(self):
        return self.ids

    def get_meta_game(self):
        restricted = self.payoffs[np.ix_(*self.ids)]
        return [restricted, -restricted]


class FixedPrecisionSolver(VDOMetaSolver):
    """Use insertion precision 10 while retaining the same MWU updates."""

    def _reinitialise(self, meta_games, avg, players):
        beta = self._beta
        try:
            self._beta = self._eta * self._num_steps / 10.0
            super()._reinitialise(meta_games, avg, players)
        finally:
            self._beta = beta


def restricted_ne(payoffs, ids):
    """Full-game half NashConv of a Nash equilibrium of the restricted game."""
    restricted = payoffs[np.ix_(*ids)]
    strategies = []
    for payoff in [restricted, -restricted.T]:
        n, m = payoff.shape
        result = linprog(
            np.r_[np.zeros(n), -1],
            A_ub=np.c_[-payoff.T, np.ones(m)],
            b_ub=np.zeros(m),
            A_eq=[np.r_[np.ones(n), 0]],
            b_eq=[1],
            bounds=[(0, None)] * n + [(None, None)],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Restricted-game LP failed: {result.message}")
        weights = np.maximum(result.x[:-1], 0)
        strategies.append(weights / weights.sum())
    x, y = strategies
    if np.max(restricted @ y) - np.min(x @ restricted) > 1e-7:
        raise RuntimeError("The restricted-game equilibrium residual exceeds 1e-7.")
    return float((np.max(payoffs[:, ids[1]] @ y) - np.min(x @ payoffs[ids[0], :])) / 2)


def run(payoffs, mode, seed, events=200, keep_duplicates=False):
    """Run until both players reach the requested number of BR events.

    Each recorded value averages the corresponding events of the two players.
    Events are measured before adding the response strategy. keep_duplicates
    stores repeated responses with inactive weight, for equivalence checks.
    """
    if mode not in BETAS and mode != "fixed_lambda10":
        raise ValueError(f"Unknown method: {mode}")
    if events < 1:
        raise ValueError("events must be positive")
    game = MatrixGame(payoffs, seed)
    solver_type = FixedPrecisionSolver if mode == "fixed_lambda10" else VDOMetaSolver
    solver = solver_type(
        eta=1 / 11,
        beta=BETAS.get(mode, 1.0),
        window_scale=0 if mode == "fp_hardbr" else 50,
        payoff_scale=1.0,
        strategy_key_fn=lambda player, action: action,
        per_player_windows=True,
    )
    player_events = [[], []]
    cached_rne = None
    while min(map(len, player_events)) < events:
        average = solver(game)
        due = list(solver.players_due)
        row_payoffs = payoffs[:, game.ids[1]] @ average[1]
        column_payoffs = -average[0] @ payoffs[game.ids[0], :]
        online = float((row_payoffs.max() + column_payoffs.max()) / 2)
        if cached_rne is None:
            cached_rne = restricted_ne(payoffs, game.ids)
        record = {"t": solver._num_steps, "online": online, "rne": cached_rne}
        for player in due:
            if len(player_events[player]) < events:
                player_events[player].append(dict(record))
        if min(map(len, player_events)) >= events:
            break
        for player in due:
            response = int(np.argmax([row_payoffs, column_payoffs][player]))
            distinct = response not in game.ids[player]
            if distinct or keep_duplicates:
                game.ids[player].append(response)
            if distinct:
                cached_rne = None
    points = []
    for index in range(events):
        points.append(
            {
                "br_count": index + 1,
                "online": float(np.mean([p[index]["online"] for p in player_events])),
                "rne": float(np.mean([p[index]["rne"] for p in player_events])),
                "learning_rounds": [p[index]["t"] for p in player_events],
            }
        )
    return {"points": points, "learning_rounds": solver._num_steps}


def run_spec(spec):
    size, mode, seed, events = spec
    raw = np.random.default_rng(0).normal(0, 1, (size, size))
    scale = max(1.0, float(abs(raw).max()))
    result = run(raw / scale, mode, seed, events)
    result.update(
        spec=list(spec),
        payoff_scale=scale,
        raw_matrix_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        training_payoffs="normalised",
        metric="normalised half NashConv",
    )
    return result


def filename(spec):
    size, mode, seed, events = spec
    return f"matrix{size}_{mode}_seed{seed}_q{events}.json.gz"
