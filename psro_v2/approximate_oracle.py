"""Approximate best-response oracles built on the exact BR oracle.

Two controlled kinds of BR error, each seeded for reproducibility:

- "sampled_target": the exact BR is computed against an empirical mixture of
  `num_samples` strategies drawn from each opponent's meta-strategy, instead
  of the meta-strategy itself (as when a BR is trained on sampled opponents).
- "epsilon": the exact BR, but at each of the responder's infostates the
  action is replaced by a uniformly random legal action with probability
  `epsilon`.
"""

import numpy as np

from open_spiel.python import policy as openspiel_policy
from psro_v2 import best_response_oracle


class ApproximateBestResponseOracle(best_response_oracle.BestResponseOracle):
    """BestResponseOracle returning deliberately approximate best responses."""

    def __init__(
        self,
        approximation="sampled_target",
        num_samples=10,
        epsilon=0.1,
        seed=None,
        game=None,
        **kwargs,
    ):
        """Initialises the oracle.

        Args:
          approximation: "sampled_target" or "epsilon" (see module doc).
          num_samples: Opponent strategies drawn per BR for "sampled_target".
          epsilon: Per-infostate probability of a random action for "epsilon".
          seed: Seed of the oracle's own random stream.
          game: The game (passed to BestResponseOracle).
          **kwargs: Forwarded to BestResponseOracle.
        """
        super().__init__(game=game, **kwargs)
        if approximation not in ("sampled_target", "epsilon"):
            raise ValueError(f"Unknown approximation: {approximation!r}")
        self._approximation = approximation
        self._num_samples = int(num_samples)
        self._epsilon = float(epsilon)
        self._rng = np.random.RandomState(seed)
        self._template = openspiel_policy.TabularPolicy(game)

    def _sample_target(self, probabilities, current_player):
        """Empirical mixture of num_samples draws from each opponent's mixture."""
        sampled = []
        for p, probs in enumerate(probabilities):
            probs = np.asarray(probs, dtype=np.float64)
            if p == current_player:
                sampled.append(probs)
                continue
            draws = self._rng.choice(
                probs.size, size=self._num_samples, p=probs / probs.sum()
            )
            sampled.append(
                np.bincount(draws, minlength=probs.size) / float(self._num_samples)
            )
        return sampled

    def _perturb(self, best_resp, player):
        """Tabular copy of the BR with epsilon-random actions for `player`."""
        perturbed = self._template.__copy__(copy_action_probability_array=True)
        for row, state in enumerate(perturbed.states):
            if state.current_player() != player:
                continue
            legal = np.flatnonzero(perturbed.legal_actions_mask[row])
            if self._rng.random_sample() < self._epsilon:
                action = self._rng.choice(legal)
            else:
                probs = best_resp.action_probabilities(state)
                action = max(legal, key=lambda a: probs.get(int(a), 0.0))
            perturbed.action_probability_array[row] = 0.0
            perturbed.action_probability_array[row, action] = 1.0
        return perturbed

    def __call__(
        self,
        game,
        training_parameters,
        strategy_sampler=None,
        using_joint_strategies=False,
        **kwargs,
    ):
        if using_joint_strategies:
            raise NotImplementedError(
                "ApproximateBestResponseOracle supports marginal meta-strategies "
                "only (sample_from_marginals=True, no rectifier)."
            )
        if self._approximation == "sampled_target":
            training_parameters = [
                [
                    dict(
                        params,
                        probabilities_of_playing_policies=self._sample_target(
                            params["probabilities_of_playing_policies"],
                            params["current_player"],
                        ),
                    )
                    for params in player_parameters
                ]
                for player_parameters in training_parameters
            ]
        new_policies = super().__call__(
            game, training_parameters, using_joint_strategies=False, **kwargs
        )
        if self._approximation == "epsilon":
            new_policies = [
                [
                    self._perturb(pol, params["current_player"])
                    for pol, params in zip(player_policies, player_parameters)
                ]
                for player_policies, player_parameters in zip(
                    new_policies, training_parameters
                )
            ]
        return new_policies
