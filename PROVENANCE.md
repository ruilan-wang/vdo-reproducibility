# Source and data provenance

The `psro_v2/` implementation is derived from Francisco Aristi Reina's
[VDO-PSRO](https://github.com/ferdos202150762/VDO-PSRO), supplied archive at commit
`fb92f91d5ba5012fcede786cf2d24585e0b9423e`. The tabular BR/evaluation and MWU
computations are retained. The paper runner explicitly selects per-player
windows. That setting is also the default in this distribution.

`vdo/experiment.py` retains the PSRO setup and per-player event ordering, using
restricted-NE terminology. It includes the exact and controlled approximate BR
paths needed by the paper; unused neural RL responders and their dependencies
are omitted. Restricted-game LP failures raise an explicit error. An
endpoint-only option supports sensitivity experiments without changing their
learning trajectory.

`vdo/matrix.py`, the supplementary runners, figure scripts and table generation
package Ruilan Wang's matrix and sensitivity experiments. Figures retain the
manuscript's method colours, labels, discount selections and uncertainty
convention. All 36 panels can be generated from numerical records. The original
paper assets are included separately. `data/figures.json` maps panels to data.

The supplied archive contains all 18 required PSRO aggregate result files and
885 corresponding per-seed records. Aggregate files are retained byte for byte;
individual seed files are compressed without changing their contents. Both
metric means and standard deviations agree with the per-seed records at every
recorded event. `data/psro_manifest.json` records archive, source and packaged
file hashes. Table 2 is generated from the same aggregates as the PSRO figures;
its independently supplied endpoint summaries are retained for verification.

Fifteen PSRO configuration files are transcribed from the supplied source,
omitting unused metadata and using RNE field names. The sampled-target
configurations for Goofspiel-4, Leduc-8 and Liar's Dice-5 are instantiated from
the 10-sample protocol documented in the source README and `games-ran.md`, with
supplied game definitions and recorded method/seed sets. The original three
configuration files are absent from the archive. Using the native dependency
versions, the instantiated settings reproduce the first 30 events for beta=50,
seed=1 in each of these three games within 1.1e-14 in native NashConv. This is
not a claim of full-run or original-file identity. `data/config_sources.json`
records each configuration's origin.

`data/supplement_manifest.json` records source and packaged checksums for the
720 supplementary runs. Packaged records retain experiment parameters, payoff
scale, numerical metrics and available learning-round indices; timestamps and
diagnostic prose are omitted. All packaged numerical inputs, configurations
and paper assets are covered by `data/checksums.json`.

The runtime dependency versions in `requirements-paco.txt` are taken from the
supplied VDO-PSRO lockfile. The default `uv.lock` pins the normalised-experiment
environment. Neither file asserts knowledge of the exact hardware or complete
runtime environment used to produce the original PSRO records.
