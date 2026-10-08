# Training data

The curated interaction dataset and its fixed training, validation, and test
assignments are available from Zenodo:

- **DOI:** [10.5281/zenodo.23158020](https://doi.org/10.5281/zenodo.23158020)

Large training files and embeddings are not stored directly in this Git
repository. After downloading the deposited data, configure their local paths
in `configs/train_example.yaml`. BaCNet reads the provided `transformed_score`
values directly and does not apply Box-Cox transformation, min-max scaling,
clipping, or thresholding during training.

In the data-construction workflow, `stitch_score` denotes the STITCH
`combined_score` divided by 1,000. Box-Cox transformation followed by min-max
scaling produces `transformed_score`, which is the value deposited on Zenodo
and consumed by the training code. `stitch_score` may be retained as provenance
metadata but is not a required training column.

Required links columns:

- `protein_id`: protein identifier in BaCNet format
- `compound_id`: chemical identifier in BaCNet format
- `transformed_score`: finite, already transformed regression target deposited on Zenodo
- `split`: one of `train`, `validation`, or `test`

The split strategy is leave-one-protein-out. Protein groups are assigned as a
unit, so a protein group must never occur in more than one split. If different
protein identifiers can represent the same amino-acid sequence, add a
`protein_group_id` column containing the sequence-equivalence group. Otherwise,
`protein_id` is used directly.

For transparency, `reference_leave_one_protein_out_split()` in
`src/training_data.py` records the original procedure: 10% of unique protein
groups are assigned to test, then 10% of the remaining groups are assigned to
validation, using seed 123 for both operations. This gives approximately
81%/9%/10% train/validation/test by protein-group count.

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
