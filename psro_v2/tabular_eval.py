"""Exact payoff evaluation for small two-or-more-player tabular games.

The value of a joint profile is sum_z chance(z) * prod_p reach_p(z) * u(z),
where reach_p(z) is the product of player p's action probabilities along the
path to terminal z. Each policy's reach vector is computed once and cached, so
every meta-game entry becomes a single weighted product over terminals.
"""

import numpy as np


class TabularEvaluator(object):
    """Caches the game tree once and evaluates policy profiles exactly."""

    def __init__(self, game):
        self._game = game
        self._num_players = game.num_players()

        self._infostates = [dict() for _ in range(self._num_players)]
        self._num_actions = game.num_distinct_actions()

        chance, returns, paths = [], [], []
        self._walk(
            game.new_initial_state(),
            1.0,
            [[] for _ in range(self._num_players)],
            chance,
            returns,
            paths,
        )
        self._chance = np.array(chance)
        self._returns = np.array(returns)
        self._weighted_returns = self._chance[:, None] * self._returns

        self._path_rows, self._path_actions = [], []
        for p in range(self._num_players):
            depth = max(len(path[p]) for path in paths)
            dummy = len(self._infostates[p])
            rows = np.full((len(paths), depth), dummy, dtype=np.int64)
            acts = np.zeros((len(paths), depth), dtype=np.int64)
            for z, path in enumerate(paths):
                for d, (row, action) in enumerate(path[p]):
                    rows[z, d] = row
                    acts[z, d] = action
            self._path_rows.append(rows)
            self._path_actions.append(acts)
        self._build_sequences(paths)

        self._cache = {}

    def _build_sequences(self, paths):
        """Sequence-form structure used for exact best-response values.

        A sequence (row, action) has id row * num_actions + action; the empty
        (root) sequence has id `root`. For each player we store the last sequence
        before every terminal, the parent sequence and depth of every infostate,
        and which actions are legal at each infostate.
        """
        num_a = self._num_actions
        self._terminal_seq, self._parent_seq, self._depth = [], [], []
        self._legal = []
        for p in range(self._num_players):
            num_rows = len(self._infostates[p])
            root = num_rows * num_a
            terminal_seq = np.full(len(paths), root, dtype=np.int64)
            parent_seq = np.full(num_rows, root, dtype=np.int64)
            depth = np.zeros(num_rows, dtype=np.int64)
            for z, path in enumerate(paths):
                parent = root
                for d, (row, action) in enumerate(path[p]):
                    parent_seq[row] = parent
                    depth[row] = d
                    parent = row * num_a + action
                terminal_seq[z] = parent
            legal = np.zeros((num_rows, num_a), dtype=bool)
            for row, state in self._infostates[p].values():
                legal[row, state.legal_actions()] = True
            self._terminal_seq.append(terminal_seq)
            self._parent_seq.append(parent_seq)
            self._depth.append(depth)
            self._legal.append(legal)

    def _best_response_value(self, player, contribution):
        """max over player's pure strategies of sum_z reach(z) * contribution(z)."""
        num_rows = len(self._infostates[player])
        root = num_rows * self._num_actions
        seq_val = np.zeros(root + 1)
        np.add.at(seq_val, self._terminal_seq[player], contribution)
        depth = self._depth[player]
        for d in range(depth.max(), -1, -1):
            rows = np.flatnonzero(depth == d)
            action_vals = seq_val[:root].reshape(num_rows, -1)[rows]
            action_vals = np.where(self._legal[player][rows], action_vals, -np.inf)
            np.add.at(seq_val, self._parent_seq[player][rows], action_vals.max(axis=1))
        return seq_val[root]

    def nash_conv(self, policies, probabilities):
        """Exact NashConv of the profile where player p mixes policies[p].

        Args:
          policies: List (one per player) of lists of policies.
          probabilities: List (one per player) of mixture weights over policies[p].

        Returns:
          sum_p (best-response value_p - value_p) of the mixed profile.
        """
        mixed_reach = [
            sum(
                w * self.reach(p, pol)
                for pol, w in zip(policies[p], probabilities[p])
                if w > 0
            )
            for p in range(self._num_players)
        ]
        total = 0.0
        for p in range(self._num_players):
            contribution = self._weighted_returns[:, p].copy()
            for q in range(self._num_players):
                if q != p:
                    contribution *= mixed_reach[q]
            total += (
                self._best_response_value(p, contribution)
                - contribution @ mixed_reach[p]
            )
        return float(total)

    def _walk(self, state, chance_prob, path, chance, returns, paths):
        if state.is_terminal():
            chance.append(chance_prob)
            returns.append(state.returns())
            paths.append([list(p) for p in path])
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                self._walk(
                    state.child(action),
                    chance_prob * prob,
                    path,
                    chance,
                    returns,
                    paths,
                )
            return
        player = state.current_player()
        key = state.information_state_string(player)
        table = self._infostates[player]
        if key not in table:
            table[key] = (len(table), state.clone())
        row = table[key][0]
        for action in state.legal_actions():
            path[player].append((row, action))
            self._walk(state.child(action), chance_prob, path, chance, returns, paths)
            path[player].pop()

    def policy_table(self, player, pol):
        """[num_infostates + 1, num_actions] action probabilities (last row = 1)."""
        return self._entry(player, pol)[0]

    def reach(self, player, pol):
        """Product of `player`'s action probabilities along each terminal path."""
        return self._entry(player, pol)[1]

    def _entry(self, player, pol):
        key = (player, id(pol))
        if key not in self._cache:
            infostates = self._infostates[player]
            table = np.ones((len(infostates) + 1, self._num_actions))
            for row, state in infostates.values():
                table[row] = 0.0
                for action, prob in pol.action_probabilities(state).items():
                    table[row, action] = prob
            reach = np.prod(
                table[self._path_rows[player], self._path_actions[player]], axis=1
            )
            self._cache[key] = (table, reach, pol)
        return self._cache[key][:2]

    def _own_infostate_reach(self, player, pol):
        """Product of `player`'s own action probabilities on the way to each row."""
        key = ("infostate_reach", player, id(pol))
        if key not in self._cache:
            table = self.policy_table(player, pol)
            num_rows = len(self._infostates[player])
            seq_reach = np.ones(num_rows * self._num_actions + 1)
            row_reach = np.ones(num_rows)
            depth = self._depth[player]
            for d in range(depth.max() + 1):
                rows = np.flatnonzero(depth == d)
                row_reach[rows] = seq_reach[self._parent_seq[player][rows]]
                seq_reach[:-1].reshape(num_rows, -1)[rows] = (
                    row_reach[rows, None] * table[rows]
                )
            self._cache[key] = (row_reach, None, pol)
        return self._cache[key][0]

    def aggregate(self, game, policies, probabilities):
        """Behaviour policy realisation-equivalent to the mixed profile.

        Exact, vectorised counterpart of policy_aggregator.PolicyAggregator: at
        each infostate, pi(a|I) = sum_j w_j R_j(I) pi_j(a|I) / sum_j w_j R_j(I),
        with R_j(I) policy j's own reach probability. Unreachable infostates get
        the uniform policy over legal actions.

        Args:
          game: The game (only used to build the returned TabularPolicy).
          policies: List (one per player) of lists of policies.
          probabilities: List (one per player) of mixture weights.

        Returns:
          A policy.TabularPolicy covering every player.
        """
        if not hasattr(self, "_tabular_template"):
            from open_spiel.python import policy as openspiel_policy

            self._tabular_template = openspiel_policy.TabularPolicy(game)

            self._tabular_rows = [
                np.array(
                    [
                        self._tabular_template.state_lookup[key]
                        for key in self._infostates[p]
                    ],
                    dtype=np.int64,
                )
                for p in range(self._num_players)
            ]
        aggregated = self._tabular_template.__copy__(copy_action_probability_array=True)
        for p in range(self._num_players):
            num_rows = len(self._infostates[p])
            numerator = np.zeros((num_rows, self._num_actions))
            denominator = np.zeros(num_rows)
            for pol, w in zip(policies[p], probabilities[p]):
                if w <= 0:
                    continue
                reach = w * self._own_infostate_reach(p, pol)
                numerator += reach[:, None] * self.policy_table(p, pol)[:-1]
                denominator += reach
            legal = self._legal[p]
            uniform = legal / legal.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mixed = np.where(
                    denominator[:, None] > 0, numerator / denominator[:, None], uniform
                )

            aggregated.action_probability_array[self._tabular_rows[p]] = mixed
        return aggregated

    def fingerprint(self, player, pol):
        """Hashable identity of the strategy `pol` plays as `player`."""
        return np.round(self.policy_table(player, pol), 9).tobytes()

    def expected_returns(self, policies):
        """Exact expected returns of the joint profile (one policy per player)."""
        reach = np.ones(len(self._chance))
        for p, pol in enumerate(policies):
            reach = reach * self.reach(p, pol)
        return reach @ self._weighted_returns


def tabular_policy_to_dict(pol):
    """policy_utils.policy_to_dict for a full-game TabularPolicy, per infostate.

    Builds the same {infostate: [(action, prob), ...]} mapping (legal actions in
    increasing order) directly from the policy table instead of visiting every
    game state.
    """
    probs = pol.action_probability_array.tolist()
    mask = pol.legal_actions_mask
    return {
        key: [(a, probs[row][a]) for a in np.flatnonzero(mask[row]).tolist()]
        for key, row in pol.state_lookup.items()
    }


def install_fast_policy_to_dict(game):
    """Routes policy_utils.policy_to_dict through tabular_policy_to_dict.

    Applies only to TabularPolicy objects covering every player of `game`;
    everything else goes to OpenSpiel's original implementation. Both the BR
    oracle and OpenSpiel's CPPBestResponsePolicy call policy_to_dict on the
    aggregated policy, which otherwise visits every game state each time.
    """
    from open_spiel.python import policy as openspiel_policy
    from open_spiel.python.algorithms import policy_utils

    original = getattr(
        policy_utils.policy_to_dict, "_original", policy_utils.policy_to_dict
    )
    all_players = list(range(game.num_players()))

    def policy_to_dict(
        player_policy, game, all_states=None, state_to_information_state=None
    ):
        if (
            isinstance(player_policy, openspiel_policy.TabularPolicy)
            and list(player_policy.player_ids) == all_players
        ):
            return tabular_policy_to_dict(player_policy)
        return original(player_policy, game, all_states, state_to_information_state)

    policy_to_dict._original = original
    policy_utils.policy_to_dict = policy_to_dict
