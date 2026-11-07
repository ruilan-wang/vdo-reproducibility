# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Meta-strategy solvers for PSRO."""

import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.algorithms import regret_matching
import pyspiel

EPSILON_MIN_POSITIVE_PROBA = 1e-8


def uniform_strategy(solver, return_joint=False):
    """Returns a Random Uniform distribution on policies.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      uniform distribution on strategies.
    """
    policies = solver.get_policies()
    policy_lengths = [len(pol) for pol in policies]
    result = [np.ones(pol_len) / pol_len for pol_len in policy_lengths]
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def softmax_on_range(number_policies):
    x = np.array(list(range(number_policies)))
    x = np.exp(x - x.max())
    x /= np.sum(x)
    return x


def uniform_biased_strategy(solver, return_joint=False):
    """Returns a Biased Random Uniform distribution on policies.

    The uniform distribution is biased to prioritize playing against more recent
    policies (Policies that were appended to the policy list later in training)
    instead of older ones.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      uniform distribution on strategies.
    """
    policies = solver.get_policies()
    if not isinstance(policies[0], list):
        policies = [policies]
    policy_lengths = [len(pol) for pol in policies]
    result = [softmax_on_range(pol_len) for pol_len in policy_lengths]
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def renormalize(probabilities):
    """Replaces all negative entries with zeroes and normalizes the result.

    Args:
      probabilities: probability vector to renormalize. Has to be one-dimensional.

    Returns:
      Renormalized probabilities.
    """
    probabilities[probabilities < 0] = 0
    probabilities = probabilities / np.sum(probabilities)
    return probabilities


def get_joint_strategy_from_marginals(probabilities):
    """Returns a joint strategy tensor from a list of marginals.

    Args:
      probabilities: list of probabilities.

    Returns:
      A flat joint strategy from a list of marginals.
    """
    res = np.ones((1,), dtype=np.float64)
    for prob in probabilities:
        res = res[..., None] @ np.asarray(prob).reshape((1,) * res.ndim + (-1,))
    return res.reshape(-1)


def nash_strategy(solver, return_joint=False):
    """Returns nash distribution on meta game matrix.

    This method only works for two player zero-sum games.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      Nash distribution on strategies.
    """
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    meta_games = [x.tolist() for x in meta_games]
    if len(meta_games) != 2:
        raise NotImplementedError(
            "nash_strategy solver works only for 2p zero-sum"
            "games, but was invoked for a {} player game".format(len(meta_games))
        )
    nash_prob_1, nash_prob_2, _, _ = lp_solver.solve_zero_sum_matrix_game(
        pyspiel.create_matrix_game(*meta_games)
    )
    result = [
        renormalize(np.array(nash_prob_1).reshape(-1)),
        renormalize(np.array(nash_prob_2).reshape(-1)),
    ]

    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def prd_strategy(solver, return_joint=False):
    """Computes Projected Replicator Dynamics strategies.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      PRD-computed strategies.
    """
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    kwargs = solver.get_kwargs()
    result = projected_replicator_dynamics.projected_replicator_dynamics(
        meta_games, **kwargs
    )
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def rm_strategy(solver, return_joint=False):
    """Computes regret-matching strategies.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, only returns marginals. Otherwise marginals as well
        as joint probabilities.

    Returns:
      PRD-computed strategies.
    """
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]
    kwargs = solver.get_kwargs()
    result = regret_matching.regret_matching(meta_games, **kwargs)
    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def mwu_strategy(solver, return_joint=False, iterations=10, eta=1 / 11):
    """Computes strategies via Multiplicative Weights Update (MWU).

    Maintains raw (unnormalized) weights internally. At each step:
      1. Probabilities are derived from raw weights (p = w / sum(w)).
      2. Expected payoffs are computed using opponent probabilities.
      3. Raw weights are updated multiplicatively: w *= exp(eta * payoff).
      4. The time-average is over probabilities (not raw weights).

    The step_callback and the returned result are always probabilities.

    Args:
      solver: GenPSROSolver instance.
      return_joint: If true, returns marginals and joint. Otherwise just marginals.
      iterations: Number of MWU update steps.
      eta: Learning rate for the multiplicative update.

    The solver kwarg `mwu_initial_weights` may be set to a callable with
    signature fn(solver) -> List[array-like], returning one probability vector
    per player. Treated as initial raw weights (already non-negative; need not
    sum to 1). If omitted, uniform weights are used.

    Returns:
      MWU time-averaged strategies (probabilities) for each player.
    """
    meta_games = solver.get_meta_game()
    if not isinstance(meta_games, list):
        meta_games = [meta_games, -meta_games]

    num_players = len(meta_games)
    meta_games = [np.array(g, dtype=np.float64) for g in meta_games]
    num_strategies = [meta_games[0].shape[p] for p in range(num_players)]

    kwargs = solver.get_kwargs()
    initial_weights_fn = kwargs.get("mwu_initial_weights", None)
    step_callback = kwargs.get("mwu_step_callback", None)
    iterations = kwargs.get("mwu_iterations", iterations)
    eta = kwargs.get("mwu_eta", eta)

    first_round = any(n <= 1 for n in num_strategies)
    if not first_round and initial_weights_fn is not None:
        raw = initial_weights_fn(solver)
        weights = [np.array(raw[p], dtype=np.float64) for p in range(num_players)]
    else:
        weights = [np.ones(n, dtype=np.float64) for n in num_strategies]

    cumulative_probs = [np.zeros(n) for n in num_strategies]

    letters = "abcdefghijklmnopqrstuvwxyz"[:num_players]

    for _ in range(iterations):

        probs = [w / np.linalg.norm(w, 1) for w in weights]

        for player_id in range(num_players):
            cumulative_probs[player_id] += probs[player_id]

        if step_callback is not None:
            step_callback(probs)

        for player_id in range(num_players):
            other_letters = [letters[q] for q in range(num_players) if q != player_id]
            subscripts = (
                letters + "," + ",".join(other_letters) + "->" + letters[player_id]
            )
            operands = [meta_games[player_id]] + [
                probs[q] for q in range(num_players) if q != player_id
            ]
            expected_payoffs = np.einsum(subscripts, *operands)
            expected_payoffs = expected_payoffs - expected_payoffs.mean()

            weights[player_id] = weights[player_id] * np.exp(eta * expected_payoffs)

    result = [cp / iterations for cp in cumulative_probs]

    if not return_joint:
        return result
    else:
        joint_strategies = get_joint_strategy_from_marginals(result)
        return result, joint_strategies


def _expected_payoffs(meta_games, probs, player_id):
    """Payoff of each of player_id's strategies against the others' mixtures."""
    num_players = len(meta_games)
    letters = "abcdefghijklmnopqrstuvwxyz"[:num_players]
    others = [q for q in range(num_players) if q != player_id]
    subscripts = (
        letters + "," + ",".join(letters[q] for q in others) + "->" + letters[player_id]
    )
    return np.einsum(subscripts, meta_games[player_id], *[probs[q] for q in others])


def _softmax(log_weights):
    z = np.exp(log_weights - np.max(log_weights))
    return z / z.sum()


class VDOMetaSolver(object):
    """Persistent MWU meta-solver with population expansion.

    The paper uses per-player BR countdowns of floor(c * ln(k_i)) learning
    rounds (at least one round for singleton populations). Only a player
    that discovers a distinct strategy is reinitialised. Its weights are
    proportional to exp(eta * t * payoff_against_average / beta).
    All players update simultaneously; the returned profile averages every
    distribution played since the first learning round.

    beta=1 gives VDO, beta>1 gives dVDO, and beta=inf gives ODO.
    beta=0 with window_scale=0 gives the FP hard-BR limit, retargeted at
    every BR event even when no distinct strategy is discovered.
    per_player_windows=False enables shared windows and joint resets;
    the supplied paper configurations all use per_player_windows=True.
    """

    def __init__(
        self,
        eta=1 / 11,
        beta=1.0,
        window_scale=50.0,
        payoff_scale=1.0,
        strategy_key_fn=None,
        per_player_windows=True,
    ):
        """Initialises the solver.

        Args:
          eta: MWU learning rate.
          beta: Discount on the reinitialisation temperature (see class doc).
          window_scale: c in the per-player countdown floor(c * ln(k_i)).
          payoff_scale: Divisor applied to meta-game payoffs; use the maximum
            absolute game payoff for normalised training, or 1 for native units.
          strategy_key_fn: Callable (player, policy) -> hashable identifying the
            pure strategy. A policy whose key is already in the population is a
            re-discovery: it gets zero weight forever and triggers no
            reinitialisation, matching "if b* not in Pi" in Algorithm 1. If None,
            every policy is treated as new.
          per_player_windows: Use separate BR countdowns and reinitialise only
            the player whose population grows, as in the paper.
        """
        self._eta = eta
        self._beta = beta
        self._window_scale = window_scale
        self._payoff_scale = payoff_scale
        self._strategy_key_fn = strategy_key_fn
        self._log_weights = None
        self._active = None
        self._keys = None
        self._avg_sum = None
        self._num_steps = 0
        self._per_player_windows = per_player_windows
        self._countdown = None

        self.players_due = None

    def _window_length(self):
        k = min(int(active.sum()) for active in self._active)
        if self._window_scale <= 0:
            return 1
        return int(np.ceil(self._window_scale * np.log(k + 1)))

    def _refractory(self, player):
        """BR countdown: floor(c * ln n), 0 while n <= 1."""
        n = int(self._active[player].sum())
        if n <= 1 or self._window_scale <= 0:
            return 0
        return int(self._window_scale * np.log(n))

    def _reinitialise(self, meta_games, avg, players):
        for p in players:
            active = self._active[p]
            log_w = np.full(active.size, -np.inf)
            if np.isinf(self._beta):
                log_w[active] = 0.0
            else:
                payoffs = _expected_payoffs(meta_games, avg, p)[active]
                if self._beta == 0:
                    best = np.isclose(payoffs, payoffs.max(), rtol=0, atol=1e-12)
                    log_w[np.flatnonzero(active)[best]] = 0.0
                else:
                    temperature = self._eta * self._num_steps / self._beta
                    log_w[active] = temperature * payoffs
            self._log_weights[p] = log_w

    def __call__(self, solver, return_joint=False):
        meta_games = solver.get_meta_game()
        if not isinstance(meta_games, list):
            meta_games = [meta_games, -meta_games]
        meta_games = [
            np.asarray(g, dtype=np.float64) / self._payoff_scale for g in meta_games
        ]
        policies = solver.get_policies()
        num_players = len(meta_games)
        sizes = [meta_games[0].shape[p] for p in range(num_players)]

        first_call = self._log_weights is None
        if first_call:
            self._log_weights = [np.zeros(0) for _ in range(num_players)]
            self._active = [np.zeros(0, dtype=bool) for _ in range(num_players)]
            self._keys = [set() for _ in range(num_players)]
            self._avg_sum = [np.zeros(0) for _ in range(num_players)]

        discovered = [False] * num_players
        for p in range(num_players):
            old = self._active[p].size
            new_active = np.zeros(sizes[p] - old, dtype=bool)
            for i in range(old, sizes[p]):
                key = (
                    self._strategy_key_fn(p, policies[p][i])
                    if self._strategy_key_fn is not None
                    else i
                )
                if key not in self._keys[p]:
                    self._keys[p].add(key)
                    new_active[i - old] = True
            discovered[p] = bool(new_active.any())
            self._active[p] = np.concatenate([self._active[p], new_active])
            self._avg_sum[p] = np.concatenate(
                [self._avg_sum[p], np.zeros(sizes[p] - old)]
            )
            self._log_weights[p] = np.concatenate(
                [self._log_weights[p], np.full(sizes[p] - old, -np.inf)]
            )

        if first_call:
            for p in range(num_players):
                self._log_weights[p][self._active[p]] = 0.0
            self._countdown = [0] * num_players
        else:
            if self._per_player_windows:

                reinit = [
                    p for p in self.players_due if discovered[p] or self._beta == 0
                ]
            elif any(discovered) or self._beta == 0:

                reinit = list(range(num_players))
            else:
                reinit = []
            if reinit:
                avg = [s / self._num_steps for s in self._avg_sum]
                self._reinitialise(meta_games, avg, reinit)
            if self._per_player_windows:
                for p in self.players_due:
                    self._countdown[p] = self._refractory(p)

        num_steps = 0
        while True:
            probs = [_softmax(log_w) for log_w in self._log_weights]
            for p in range(num_players):
                self._avg_sum[p] += probs[p]
            self._num_steps += 1
            for p in range(num_players):
                self._log_weights[p] = self._log_weights[p] + self._eta * (
                    _expected_payoffs(meta_games, probs, p)
                )
            num_steps += 1
            if self._per_player_windows:
                self._countdown = [c - 1 for c in self._countdown]
                due = [p for p in range(num_players) if self._countdown[p] <= 0]
                if due:
                    self.players_due = due
                    break
            elif num_steps >= self._window_length():
                self.players_due = list(range(num_players))
                break

        result = [s / self._num_steps for s in self._avg_sum]
        if not return_joint:
            return result
        else:
            joint_strategies = get_joint_strategy_from_marginals(result)
            return result, joint_strategies


def due_players_selector(meta_solver, base_selector):
    """Training-strategy selector restricting PSRO's BRs to due players.

    Wraps a selector (e.g. strategy_selectors.probabilistic) and empties the
    selection of every player not in `meta_solver.players_due`, so PSRO computes
    no BR for them this iteration. Solvers without `players_due` (e.g. "nash")
    leave the selection unchanged.

    Args:
      meta_solver: The meta-strategy solver instance used by PSRO.
      base_selector: Selector with signature (solver, number_policies_selected).

    Returns:
      A selector with the same signature as base_selector.
    """

    def select(solver, number_policies_selected=1):
        used_policies, used_indexes = base_selector(solver, number_policies_selected)
        due = getattr(meta_solver, "players_due", None)
        if due is not None:
            for p in range(len(used_policies)):
                if p not in due:
                    used_policies[p], used_indexes[p] = [], []
        return used_policies, used_indexes

    return select


META_STRATEGY_METHODS = {
    "uniform_biased": uniform_biased_strategy,
    "uniform": uniform_strategy,
    "nash": nash_strategy,
    "prd": prd_strategy,
    "rm": rm_strategy,
    "mwu": mwu_strategy,
}
