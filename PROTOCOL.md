# Experimental protocol

## Learning and population growth

- Two-player zero-sum games; both players take simultaneous MWU updates.
- Learning rate `eta=1/11`, except the Kuhn learning-rate sweep.
- Each player has its own BR countdown `floor(50*ln(k_i))`; at least one
  learning round is performed when the population is a singleton.
- Only a player whose response is a distinct strategy is reinitialised.
  A repeated strategy has no additional active population weight.
- BR targets use the opponent's average over all preceding learning rounds.
  Weight reconstruction uses this average and precision `eta*t/beta`.
- `beta=1`: VDO; finite `beta>1`: dVDO; `beta=inf`: ODO uniform reset.
- The FP hard-BR limit uses `beta=0` and one-round response intervals. It
  retargets the average at each event, including rediscoveries.
- Figure 10 also uses insertion precision fixed at 10, with the same MWU updates.

The horizontal axis counts BR events **per player**, including rediscoveries.
The q-th plotted value averages the measurements at each player's q-th event,
before inserting the resulting response. These two events need not occur at
the same learning round. Execution stops when both players have q events.

## Payoffs and evaluation

Let `M` be the maximum absolute payoff. All reported exploitabilities use
`NashConv/(2*M)` in a two-player zero-sum game. In these units the largest
possible exploitability is one.

- **Online exploitability:** evaluate the average profile produced by learning.
- **Restricted-NE exploitability (RNE):** solve the current restricted game for
  a Nash equilibrium and evaluate that profile against full-game best responses.

For extensive-form games, the policy mixture is converted to a
realisation-equivalent behaviour policy using each constituent policy's own
reach probabilities. This is built into the evaluator; there is no separate
`realisation=True` switch.

| Experiment family | Training payoffs | Stored metrics | Display conversion |
|---|---|---|---|
| PSRO exact/epsilon/sampled-target | Native units | Full native NashConv | Divide by `2*M` |
| Kuhn/Leduc sensitivity runs | Divided by `M` | Normalised half NashConv | None |
| Gaussian matrices | Divided by maximum absolute matrix entry | Half NashConv of the normalised game | None |

`M=2` for Kuhn; `M=13` for Leduc and Leduc-8; `M=1` for Goofspiel-4, Blotto
and Liar's Dice-5. Rescaling only the displayed exploitability does not
rescale the learning updates. Therefore the two training conventions are
kept explicit in configuration files and result metadata.

## Response oracles

All PSRO payoff matrices and reported exploitabilities are evaluated exactly.
The robustness settings change the response oracle:

1. `exact`: exact BR to the opponent mixture.
2. `eps`: compute an exact BR, then replace the action at each responder
   information state with a uniformly sampled legal action with probability 0.1.
3. `sampled`: sample 10 opponent policies from the mixture and compute an exact
   BR to their empirical mixture.

The third setting is sampled-target response computation, not finite-rollout
payoff estimation. Both controlled approximate oracles use a separate random
stream with seed `seed+10000`.

## Seeds, matrices and uncertainty

Kuhn, Leduc and Blotto use seeds 1–10. The larger PSRO games use seeds 1–5.
Initial extensive-form policies select a random legal action at each information
state. Matrix initial actions use a NumPy `RandomState` with the run's seed.

For each matrix dimension `n in {25,50,100}`, the payoff matrix is a single
fixed draw `default_rng(0).normal(0,1,(n,n))`, divided by its maximum absolute
entry. The 10 runs vary initial actions, not the matrix. Results at different
sizes are therefore not an ensemble over independently drawn matrices.

All ordinary endpoints are event 200 per player; Figure 10 uses event 3000.
Kuhn sensitivity uses `eta in {0.01,0.05,1/11,0.2,0.5}` and discounts
`{1,2,5,10,20,50}` plus ODO. Leduc uses the same discounts at `eta=1/11`.

Shading/error bars are `1.96 * std(ddof=0) / sqrt(number_of_seeds)`.
For logarithmic sensitivity and short matrix plots, values below `1e-4` are
drawn at `1e-4`; stored numerical values are unchanged. PSRO trajectory plots
use unmodified means and confidence intervals on logarithmic axes; nonpositive
values caused by floating-point roundoff cannot be displayed on those axes.
Table 2 shows values
below `1e-9` as approximately zero and otherwise uses three significant figures.
The selected dVDO discount is used for both online and RNE values. Exact Kuhn
uses beta=2 as in the selected trajectory plot; beta=2 and beta=5 have the same
displayed online value at three significant figures.

Small floating-point differences can change which tied BR is selected and hence
later populations. A fresh simulation is not guaranteed to be bit-for-bit
identical to the supplied records. Replotting the records preserves their
numerical results. Each fresh checkpoint records its configuration, source hash,
Python version and numerical package versions.
