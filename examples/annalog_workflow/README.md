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

Source paths in `run_manifest.csv` are relative to a data root that contains
the two experiment directories:

```text
data-root/
├── 251209_bacnet_outputs_1st_generation/
└── 260106_2nd_generation/
```

## Run first-generation generation and BaCNet ranking

The top-level wrapper runs the executable portion from Mianserin through
first-generation medium/far generation, chemical filtering, three compound
embeddings, PBP1A BaCNet inference, and ranking:

```bash
python src/run_annalog_bacnet.py \
  --config configs/annalog_bacnet_example.yaml
```

Set the local ANNalog checkpoint, vocabulary, and Chemical Checker parameter
paths in the configuration before running. `--resume` reuses completed stages,
and `--dry-run` prints all cross-environment commands without executing them.

## Rebuild the combined table

```bash
python src/aggregate_annalog_results.py \
  --manifest examples/annalog_workflow/run_manifest.csv \
  --selection examples/annalog_workflow/selection_manifest.csv \
  --data-root /path/to/ANNalog-results \
  --output examples/annalog_workflow/annalog_candidates.csv \
  --summary-output examples/annalog_workflow/run_summary.csv
```
