# ANNalog expansion and BaCNet prioritization

This directory records the two-stage analog-expansion workflow underlying the
reported Figure 4 analysis. ANNalog generation and BaCNet scoring require
separate software environments; the CSV manifests provide the stable interface
between them.

## Reported scope

Iteration 1 starts from mianserin and includes the medium and far ANNalog runs.
After chemical filtering, the remaining compounds are ranked by BaCNet score.
Complex-structure prediction is then used as an independent, complementary
prioritization step for top-ranked compounds. The following three compounds are
used as parents for iteration 2:

- `1st_far_004`
- `1st_medium_001`
- `1st_medium_003`

For iteration 2, the reported analysis includes medium and far runs for
`1st_far_004` and `1st_medium_001`, and the medium run for
`1st_medium_003`.

## Selection interpretation

The complex-structure calculations are used only to prioritize candidates that
already have high BaCNet scores. Predictions were generated with Boltz-2
version 2.1.1. The structural criterion was qualitative: candidates showing a
clear protein-ligand association in the predicted complex on visual inspection
were selected.

## Chemical filters

The aggregation script applies the criteria used in the reported workflow:

- PAINS must be `False`.
- SA score must be less than or equal to 5.
- QED must be greater than or equal to 0.5.

Every generated compound remains in the combined table. Failed compounds have
`filter_pass=false`, an explicit `filter_failure_reason`, and no fabricated
BaCNet score.

## Files

- `run_manifest.csv`: generation settings, parents, and source tables for the
  seven reported runs.
- `selection_manifest.csv`: first-to-second iteration parent selection.
- `annalog_candidates.csv`: generated long-format output containing filter
  results, BaCNet scores, ranks, and selection status.
- `run_summary.csv`: per-run generation, filter, scoring, and selection counts.
- `published_runs/<run_id>/`: date-independent filter and BaCNet score tables
  for all seven reported runs. The tables contain only the columns consumed by
  the aggregation script; RDKit object representations and machine-specific
  paths are omitted.


## Install the ANNalog version used in the reported analysis

The reported analysis used ANNalog 0.5 from the official ANNalog repository at
the following commit:

```text
https://github.com/DVNecromancer/ANNalog.git
0b2c21783749b8d1296efbb0263900c94a2c1a6f
```

ANNalog 0.5 is not installed from PyPI by `environments/annalog.yml`. Create the
recorded dependency environment, clone the upstream repository, check out the
pinned commit, and install its `annalog_package` subdirectory:

The environment file mirrors the upstream `seq2seq_environment.yml`, including
PyTorch 1.9.1 and torchtext 0.10.1. The latter is required because ANNalog 0.5
imports the legacy `Field` API from `torchtext.legacy`.

```bash
conda env create -f environments/annalog.yml

git clone https://github.com/DVNecromancer/ANNalog.git /path/to/ANNalog
git -C /path/to/ANNalog checkout --detach \
  0b2c21783749b8d1296efbb0263900c94a2c1a6f

conda run --no-capture-output -n annalog \
  python -m pip install --no-deps -e /path/to/ANNalog/annalog_package
```

`--no-deps` preserves the dependency versions recorded in
`environments/annalog.yml`; in particular, that environment supplies
`partialsmiles==2.0`.

Verify the installed version and source location:

```bash
conda run -n annalog python - <<'PY'
from pathlib import Path
import annalog
from importlib.metadata import version

print("version:", version("annalog"))
print("module:", Path(annalog.__file__).resolve())
PY
```

The checkpoint and vocabulary used by the reported analysis are stored in the
repository-level `ckpt_and_vocab` directory at that commit, rather than inside
`annalog_package`:

```text
/path/to/ANNalog/ckpt_and_vocab/Lev_extended.pt
/path/to/ANNalog/ckpt_and_vocab/stereo_experiment_vocab.pkl
```

Their expected SHA-256 digests are:

```text
9e22e755b5d47da3c678bc19073e4f4af12ae061c91a7e5382cea67e30fad97b  Lev_extended.pt
a70ac489ae25d8a10584e640a3dfbf476e5a78eab48fc76e10dad75202bbcdff  stereo_experiment_vocab.pkl
```

Verify the files before running the workflow:

```bash
sha256sum \
  /path/to/ANNalog/ckpt_and_vocab/Lev_extended.pt \
  /path/to/ANNalog/ckpt_and_vocab/stereo_experiment_vocab.pkl
```

Finally, replace `/path/to/ANNalog` in
`configs/annalog_bacnet_example.yaml` with the checkout path. Also set the
Chemical Checker parameter directory in that configuration.

## Run generation and BaCNet ranking

The top-level wrapper runs the executable portion from Mianserin through
medium/far generation, chemical filtering, three compound
embeddings, PBP1A BaCNet inference, and ranking:

```bash
python src/run_annalog_bacnet.py \
  --config configs/annalog_bacnet_example.yaml
```

Set the pinned ANNalog checkpoint and vocabulary paths described above and the
Chemical Checker parameter path in the configuration before running. `--resume`
reuses completed stages, and `--dry-run` prints all cross-environment commands
without executing them.

## Rebuild the combined table

The complete historical two-generation result table is retained in
`annalog_candidates.csv`. All seven reported runs can be re-aggregated without
an external experiment directory:

```bash
python src/aggregate_annalog_results.py \
  --manifest examples/annalog_workflow/run_manifest.csv \
  --selection examples/annalog_workflow/selection_manifest.csv \
  --data-root examples/annalog_workflow \
  --output outputs/annalog_candidates.csv \
  --summary-output outputs/annalog_run_summary.csv
```
