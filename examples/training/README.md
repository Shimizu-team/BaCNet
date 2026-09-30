# Training data placeholders

The real training links and embeddings are intentionally not included yet.
`target_bct` must be prepared by the separate training-data construction
protocol. BaCNet training reads this value directly and does not apply Box-Cox
transformation, min-max scaling, clipping, or thresholding.

Required links columns:

- `protein_id`: key in the ESM-2 embedding dictionary
- `compound_id`: common key in all three chemical embedding dictionaries
- `target_bct`: finite, already transformed regression target
- `split`: one of `train`, `validation`, or `test`

Optional metadata columns such as `pair_id`, `tax_id`, and `species` are kept
for traceability. Split assignment should be frozen before training.

Embedding files must be PyTorch-serialized dictionaries with CPU-compatible
values and the following dimensions:

- protein ESM-2: 5,120
- Morgan fingerprint: 1,024
- Chemical Checker A1-A5/B1-B5: 1,280
- ChemBERTa CLS: 384

Run a test-only smoke training without real data:

```bash
python src/train.py --config configs/train_smoke.yaml --synthetic
```
