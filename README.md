# Virtual Double Oracle

Code and numerical results for **Virtual Double Oracle: Faster Convergence by
Leveraging Strategies That Were Watching All Along**, by Ruilan Wang, Francisco
Aristi Reina and Katerina Papadaki.

The implementation includes Francisco Aristi Reina's VDO-PSRO meta-solver,
exact tabular evaluator and controlled approximate-response oracles. Experiments
run on CPUs; a GPU, JAX and PyTorch are not required.

## Quick start

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run python reproduce.py verify-data
uv run python reproduce.py smoke
uv run python reproduce.py figures
```

The final command generates **all 36 panels in Figures 1–13 and Table 2** from
included numerical records. PNG/PDF figures, plotted PSRO values, supplementary
endpoint summaries and CSV/LaTeX table rows are written to `outputs/figures/`.
Replotting does not run experiments. With pip instead, create a Python 3.12
environment and install `requirements.txt`, then use `python` in place of
`uv run python`.

## Paper figures and data

| Paper figure | Content | Included numerical records |
|---|---|---|
| 1–3 | Exact, epsilon-perturbed and sampled-target PSRO | All game/oracle trajectories and seeds |
| 4 | Kuhn learning-rate sensitivity | All sensitivity runs |
| 5 | Exact-PSRO discount trajectories | All discounts and seeds |
| 6–7 | Matrix trajectories and baseline comparisons | All matrix runs |
| 8 | Discount sensitivity | All six panels |
| 9 | Matrix-size sensitivity | All matrix runs |
| 10 | Long-horizon matrix experiment | All long-horizon runs |
| 11–13 | PSRO on additional games | All game/oracle trajectories and seeds |
| Table 2 | PSRO endpoint means | Same PSRO records as the trajectory figures |

The data contain 18 PSRO game/oracle aggregates with 885 individual seed records,
and 720 normalised-training supplementary runs. All PSRO aggregates can be
reconstructed from the seed records. DO is included in Table 2 but excluded from
the figure curves. Figure 8b uses the exact-Goofspiel PSRO record; its native
payoff scale is one.

The plotting code uses the paper's colours, metric units, confidence intervals
and selected discounts. Numerical PSRO plots are redrawn as vector figures;
they need not be pixel-identical to the manuscript's original rendered assets.
Those 36 original assets are also included and can be copied without redrawing:

```bash
uv run python reproduce.py paper-assets
```

`export-paper` is an alias for numerical figure generation with a separate
default output directory. `data/figures.json` maps every panel to its data.

## Run experiments

One normalised matrix experiment:

```bash
uv run python reproduce.py matrix --size 25 --mode dvdo_b5 --seed 1 --events 200
```

All 720 normalised runs used by Figures 4 and 6–10, followed by plotting:

```bash
uv run python reproduce.py supplement --workers 4
uv run python reproduce.py figures --data outputs/runs/supplement --output outputs/fresh-figures
```

Use `--family matrix` or `--family efg` to run a subset. The full suite is much
slower than replotting; runtime depends on the game, population growth and CPU.
This command retains the supplied PSRO records for the PSRO panels and Figure 8b.

For native-payoff PSRO, use the dependency versions in the supplied VDO-PSRO
lockfile in a separate Python 3.12 environment:

```bash
uv venv .venv-paco --python 3.12
uv pip install --python .venv-paco/bin/python -r requirements-paco.txt
.venv-paco/bin/python reproduce.py psro --config configs/psro/exact/kuhn.json --output outputs/psro/exact/kuhn --workers 4
.venv-paco/bin/python reproduce.py plot-psro --results outputs/psro/exact/kuhn/all_results.json --game kuhn --selected-beta 2 --output outputs/psro/exact/kuhn/figure
```

To run all 18 PSRO configurations (Bash):

```bash
for oracle in exact eps sampled; do
  for game in kuhn leduc goofspiel4 blotto poker_4r2s liars_dice_5; do
    .venv-paco/bin/python reproduce.py psro --config "configs/psro/$oracle/$game.json" --output "outputs/psro/$oracle/$game" --workers 4 || exit 1
  done
done
uv run python reproduce.py figures --data outputs/runs/supplement --psro-data outputs/psro --output outputs/fresh-figures
```

The final command requires both complete suites. `--psro-data` supplies all
PSRO figures, Figure 8b and Table 2; it never falls back to archived PSRO values.
The paper's selected discounts remain fixed for comparison with fresh runs.

The PSRO runner accepts `--events`, `--seeds` and `--methods` for small runs:

```bash
.venv-paco/bin/python reproduce.py psro --config configs/psro/sampled/kuhn.json --output outputs/smoke-sampled --events 10 --seeds 1 --methods vdo_beta_1
```

Native-payoff PSRO and normalised supplementary training are distinct settings;
see [PROTOCOL.md](PROTOCOL.md). Fresh simulation need not reproduce every
trajectory bit for bit, because numerical differences can affect tied responses.
Replotting the included records preserves their values.

Checkpoints are written atomically. Reusing an output directory with a different
configuration, source version or environment raises an error. Use a separate
directory for each experiment specification. Completed seed runs can be reused;
an interrupted seed restarts from its beginning.

## Layout

- `reproduce.py`: command-line entry point.
- `psro_v2/`: VDO-PSRO solver, response oracles and exact evaluator.
- `vdo/`: experiment runners, figure generation and table generation.
- `configs/psro/`: explicit native-payoff game/oracle configurations.
- `data/`: numerical records, parameter selections and checksums.
- `paper_assets/`: the panels used in the manuscript.
- `tests/`: numerical equivalence, evaluation and checkpoint checks.

See [PROVENANCE.md](PROVENANCE.md) for source attribution and configuration
origins. The code is licensed under [Apache-2.0](LICENSE); existing third-party
attribution is retained in [NOTICE](NOTICE) and the applicable source files.
