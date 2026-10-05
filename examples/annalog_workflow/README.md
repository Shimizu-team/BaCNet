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

The first-generation near run and `iteration2_medium003_far` were exploratory
runs that were not included in the reported Figure 4 analysis. They are listed
in `excluded_runs.csv` but are not required by the aggregation command.

## Selection interpretation

The complex-structure calculations are used only to prioritize candidates that
already have high BaCNet scores. A structurally plausible predicted pose is not
experimental confirmation of binding, target engagement, or mechanism of
action. Before the final release, `selection_manifest.csv` must be updated with
the exact structure-prediction software/version, replicate or seed settings,
and any quantitative or visual criteria actually used. Criteria that were not
defined at the time of analysis must not be added retrospectively.

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
- `excluded_runs.csv`: exploratory runs outside the reported Figure 4 scope.
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

## Rebuild the combined table

```bash
python src/aggregate_annalog_results.py \
  --manifest examples/annalog_workflow/run_manifest.csv \
  --selection examples/annalog_workflow/selection_manifest.csv \
  --data-root /path/to/ANNalog-results \
  --output examples/annalog_workflow/annalog_candidates.csv \
  --summary-output examples/annalog_workflow/run_summary.csv
```

The script removes non-portable serialized RDKit object strings, normalizes
legacy identifiers such as `tensor(450)`, validates one-to-one compound IDs,
retains filter failures, and assigns BaCNet ranks independently within each run.

## ANNalog provenance

The experiments were based on upstream ANNalog commit
`0b2c21783749b8d1296efbb0263900c94a2c1a6f`, with local compatibility and CSV
output changes. The final fork/release tag and a machine-readable patch against
the upstream commit will be added when the publication release is prepared.
