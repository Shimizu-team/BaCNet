# Architecture-ablation study

This workflow reproduces the BaCNet baseline and the five alternative
architectures reported in Supplementary Table 6.

## Data contract

The published `train_links.csv`, `validation_links.csv`, and `test_links.csv`
files are the exact CPI pairs used for the reported experiment. The training
program reads these files directly and does not sample, reshuffle between
splits, or recreate the split.

The deposited dataset is available from Zenodo:

- **DOI:** [10.5281/zenodo.23241408](https://doi.org/10.5281/zenodo.23241408)

The files contain 4,000,000 training pairs, 200,000 validation pairs, and
200,000 test pairs.

The distributed files use these columns:

- `ProteinID`: protein identifier in BaCNet format
- `ChemID`: chemical identifier in BaCNet format
- `transformed_score`: prepared regression target used directly by BaCNet

The column mapping is explicit in `configs/ablation.yaml`. The program adds the
canonical internal names and split labels based on the source filename.

## Architectures

All six models consume the same 7,808-dimensional input in the order ESM-2
(5,120), Morgan fingerprint (1,024), Chemical Checker A1-A5/B1-B5 (1,280), and
ChemBERTa CLS (384).

1. `baseline`: raw concatenation followed by the original MLP
2. `residual_mlp`: raw concatenation followed by a residual MLP
3. `projected_concat`: per-modality projections followed by concatenation
4. `projected_residual_fusion`: protein projection plus the mean chemical residual
5. `projected_gated_fusion`: softmax-gated sum of projected modalities
6. `projected_concat_residual_mlp`: projected concatenation followed by a residual MLP

The baseline is the reference model; items 2-6 are the five alternative
architectures requested for the ablation study.

## Single-run seed and reproducibility

Each architecture was trained once using the fixed model seed `123`. Before
each architecture is constructed, the Python, NumPy, PyTorch, CUDA, and
DataLoader random states are reset to this seed. This ensures a reproducible
single-run comparison using the same supplied data and training order.

Every architecture is trained for exactly 50 epochs. The checkpoint with the lowest validation loss across the 50 epochs is
then loaded for held-out test evaluation. Accordingly, `best_epoch` in
`reported_results.csv` records the epoch of the selected validation-best
checkpoint.

## Run the experiment

Replace the placeholders in `configs/ablation.yaml`, or provide all paths on
the command line:

```bash
accelerate launch src/train_ablation.py \
  --config configs/ablation.yaml \
  --train-links data/ablation/train_links.csv \
  --validation-links data/ablation/validation_links.csv \
  --test-links data/ablation/test_links.csv \
  --protein-embeddings data/embeddings/protein_esm2.pt \
  --morgan-embeddings data/embeddings/morgan_fingerprint.pt \
  --cc-embeddings data/embeddings/chemical_checker.pt \
  --chemberta-embeddings data/embeddings/chemberta-2.pt \
  --output-dir outputs/ablation
```

The output contains input-file checksums, the resolved configuration, a split
summary, per-architecture metrics and checkpoints, held-out predictions, and
the combined `ablation_results.csv` table.

For a CPU-only software-path test that does not reproduce scientific results:

```bash
python src/train_ablation.py \
  --config configs/ablation_smoke.yaml \
  --synthetic
```

## Reported single-run results

| Architecture | PCC | SCC | R2 | MSE |
|---|---:|---:|---:|---:|
| `baseline` | 0.701363 | 0.674018 | 0.471888 | 0.041619 |
| `residual_mlp` | 0.695089 | 0.667126 | 0.448253 | 0.043481 |
| `projected_concat` | 0.691007 | 0.661418 | 0.460600 | 0.042508 |
| `projected_residual_fusion` | 0.657073 | 0.621553 | 0.413813 | 0.046195 |
| `projected_gated_fusion` | 0.654217 | 0.611045 | 0.411924 | 0.046344 |
| `projected_concat_residual_mlp` | 0.697348 | 0.665630 | 0.466045 | 0.042079 |
