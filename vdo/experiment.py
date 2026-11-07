"""PSRO experiment setup and event recording, based on Francisco Aristi Reina's VDO-PSRO implementation."""

import json
import os
import time
import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_environment
from psro_v2 import approximate_oracle
from psro_v2 import best_response_oracle
from psro_v2 import meta_strategies
from psro_v2 import psro_v2
from psro_v2 import tabular_eval
from psro_v2 import strategy_selectors


class ExactPayoffPSROSolver(psro_v2.PSROSolver):
    """PSROSolver whose meta-game entries are exact expected returns.

    Only the payoff estimator is replaced (Monte Carlo -> exact tree
    evaluation); the PSRO loop itself is inherited unchanged.
    """

    def __init__(self, game, *args, evaluator=None, **kwargs):
        self._evaluator = evaluator or tabular_eval.TabularEvaluator(game)
        super().__init__(game, *args, **kwargs)

    def sample_episodes(self, policies, num_episodes):
        return self._evaluator.expected_returns(policies)


def _random_pure_policy(game, rng):
    """Tabular policy playing one uniformly drawn legal action per infostate."""
    pol = policy.TabularPolicy(game)
    for row, mask in enumerate(pol.legal_actions_mask):
        probs = np.zeros_like(pol.action_probability_array[row])
        probs[rng.choice(np.flatnonzero(mask))] = 1.0
        pol.action_probability_array[row] = probs
    return pol


def _init_br_responder(
    game, n_players, evaluator, initial_policy="uniform", seed=None, approximate_br=None
):
    random_policy = policy.TabularPolicy(game)
    oracle_kwargs = dict(
        game=game, policy=random_policy, aggregator=evaluator.aggregate
    )
    if approximate_br:
        oracle = approximate_oracle.ApproximateBestResponseOracle(
            approximation=approximate_br["type"],
            num_samples=approximate_br.get("num_samples", 10),
            epsilon=approximate_br.get("epsilon", 0.1),
            seed=None if seed is None else seed + 10000,
            **oracle_kwargs,
        )
    else:
        oracle = best_response_oracle.BestResponseOracle(**oracle_kwargs)
    if initial_policy == "random_pure":
        rng = np.random.RandomState(seed)
        agents = [_random_pure_policy(game, rng) for _ in range(n_players)]
    else:
        agents = [random_policy.__copy__() for _ in range(n_players)]
    return (oracle, agents)


def _zero_sum_nash(meta_games):
    """Restricted-game NE of a 2p zero-sum meta-game via HiGHS (scipy).

    Each player p solves max_{x, v} v  s.t.  x^T M_p >= v, sum(x) = 1, x >= 0,
    with M_0 = meta_games[0] and M_1 = meta_games[1]^T.
    """
    from scipy.optimize import linprog

    result = []
    for payoff in (meta_games[0], np.asarray(meta_games[1]).T):
        n, m = payoff.shape
        cost = np.zeros(n + 1)
        cost[-1] = -1.0
        a_ub = np.hstack([-payoff.T, np.ones((m, 1))])
        a_eq = np.hstack([np.ones((1, n)), np.zeros((1, 1))])
        res = linprog(
            cost,
            A_ub=a_ub,
            b_ub=np.zeros(m),
            A_eq=a_eq,
            b_eq=[1.0],
            bounds=[(0, None)] * n + [(None, None)],
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Restricted-game LP failed: {res.message}")
        result.append(meta_strategies.renormalize(res.x[:n]))
    return result


def _compute_rne_exploitability(solver, evaluator):
    """Full-game exploitability of the restricted-game NE (2p zero-sum)."""
    nash_probabilities = _zero_sum_nash(solver.get_meta_game())
    return evaluator.nash_conv(solver.get_policies(), nash_probabilities)


def _parse_beta(beta):
    """Config beta: number, or null / "inf" for ODO's uniform reset."""
    if beta is None or beta == "inf":
        return np.inf
    return float(beta)


def _run_br_event_loop(
    solver, evaluator, n_players, max_br, track_rne, start_time, endpoint_only=False
):
    """Record each player's BR events before population expansion.

    The q-th record averages the two players' q-th events, which may occur
    at different learning rounds. Stop when both players have max_br events.
    Values are full NashConv in native game units. endpoint_only computes
    diagnostics only at the final events and does not alter learning."""
    if endpoint_only:
        return _endpoint_loop(
            solver, evaluator, n_players, max_br, track_rne, start_time
        )
    meta_solver = solver._meta_strategy_method
    events = [[] for _ in range(n_players)]
    while min((len(e) for e in events)) < max_br:
        due = getattr(meta_solver, "players_due", None) or range(n_players)
        event = {
            "elapsed_time": time.time() - start_time,
            "exploitability": evaluator.nash_conv(
                solver.get_policies(), solver.get_meta_strategies()
            ),
        }
        if track_rne:
            event["rne_exploitability"] = _compute_rne_exploitability(solver, evaluator)
        for p in due:
            if len(events[p]) < max_br:
                events[p].append(event)
        counts = [len(e) for e in events]
        print(
            f"  br_events={counts}  t={event['elapsed_time']:.2f}s  expl={event['exploitability']:.6f}"
        )
        if min(counts) >= max_br:
            break
        solver.iteration()
    records = []
    for i in range(max_br):
        per_player = [events[p][i] for p in range(n_players)]
        record = {
            "br_count": i + 1,
            "elapsed_time": max((e["elapsed_time"] for e in per_player)),
            "exploitability": float(np.mean([e["exploitability"] for e in per_player])),
        }
        if track_rne:
            record["rne_exploitability"] = float(
                np.mean([e["rne_exploitability"] for e in per_player])
            )
        records.append(record)
    return records


def run_single(cfg, exp_cfg, seed, endpoint_only=False):
    """Run one PSRO experiment and return its per-BR-event records."""
    np.random.seed(seed)
    n_players = cfg["n_players"]
    training_strategy_selector = cfg.get("training_strategy_selector", "probabilistic")
    meta_strategy_method = exp_cfg["meta_strategy_method"]
    game_params = cfg.get("game_params", {"players": n_players})
    game = pyspiel.load_game_as_turn_based(cfg["game_name"], game_params)
    env = rl_environment.Environment(game)
    evaluator = tabular_eval.TabularEvaluator(game)
    tabular_eval.install_fast_policy_to_dict(game)
    if meta_strategy_method == "vdo":
        payoff_scale = (
            max(abs(game.min_utility()), abs(game.max_utility()))
            if exp_cfg.get("normalize_payoffs", False)
            else 1.0
        )
        meta_strategy_method = meta_strategies.VDOMetaSolver(
            eta=exp_cfg.get("vdo_eta", 1 / 11),
            beta=_parse_beta(exp_cfg.get("vdo_beta", 1.0)),
            window_scale=exp_cfg.get("vdo_window_scale", 50.0),
            payoff_scale=payoff_scale,
            strategy_key_fn=evaluator.fingerprint,
            per_player_windows=True,
        )
        training_strategy_selector = meta_strategies.due_players_selector(
            meta_strategy_method,
            strategy_selectors.TRAINING_STRATEGY_SELECTORS[training_strategy_selector],
        )
    if cfg.get("oracle_type", "BR") != "BR":
        raise ValueError(
            "This distribution supports exact and controlled approximate BR oracles."
        )
    oracle, agents = _init_br_responder(
        game,
        n_players,
        evaluator,
        cfg.get("initial_policy", "uniform"),
        seed,
        cfg.get("approximate_br"),
    )
    solver = ExactPayoffPSROSolver(
        env.game,
        oracle,
        evaluator=evaluator,
        initial_policies=agents,
        training_strategy_selector=training_strategy_selector,
        rectifier=cfg.get("rectifier", ""),
        sims_per_entry=cfg.get("sims_per_entry", 1),
        number_policies_selected=cfg.get("number_policies_selected", 1),
        meta_strategy_method=meta_strategy_method,
        sample_from_marginals=True,
        symmetric_game=cfg.get("symmetric_game", False),
    )
    return _run_br_event_loop(
        solver,
        evaluator,
        n_players,
        cfg["max_br"],
        cfg.get("rne", False),
        time.time(),
        endpoint_only=endpoint_only,
    )


def _endpoint_loop(solver, evaluator, n_players, max_br, track_rne, start_time):
    """Record the final per-player events without intermediate diagnostics."""
    meta = solver._meta_strategy_method
    counts = [0] * n_players
    final = [None] * n_players
    while min(counts) < max_br:
        due = getattr(meta, "players_due", None) or range(n_players)
        if any(counts[p] == max_br - 1 for p in due):
            event = {
                "elapsed_time": time.time() - start_time,
                "exploitability": evaluator.nash_conv(
                    solver.get_policies(), solver.get_meta_strategies()
                ),
            }
            if track_rne:
                event["rne_exploitability"] = _compute_rne_exploitability(
                    solver, evaluator
                )
        for p in due:
            if counts[p] < max_br:
                counts[p] += 1
                if counts[p] == max_br:
                    final[p] = dict(event)
        if min(counts) >= max_br:
            break
        solver.iteration()
    result = {
        "br_count": max_br,
        "elapsed_time": max(e["elapsed_time"] for e in final),
        "exploitability": float(np.mean([e["exploitability"] for e in final])),
    }
    if track_rne:
        result["rne_exploitability"] = float(
            np.mean([e["rne_exploitability"] for e in final])
        )
    return [result]
