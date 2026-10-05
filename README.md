# A Bacteria-Centric Deep Learning Framework for Prioritizing Antibacterial Compounds against Multidrug-Resistant Bacteria Using BaCNet


A computational pipeline for **compound–bacterial protein interaction (CPI) prediction** using
chemical embeddings and machine learning models.


This repository provides tools for:

- Generating chemical embeddings
- Running inference for CPI prediction
- Scoring compounds
- Searching and ranking candidate molecules
- Reconstructing the reported two-stage ANNalog/BaCNet prioritization workflow

---

## Installation

### Core BaCNet environment

```bash
conda env create -f environments/bacnet.yml
conda activate bacnet
```

The supplied YAML records the version-pinned environment used for BaCNet
training, inference, and the architecture-ablation study. It was exported from
a Linux GPU environment using Python 3.9.18, PyTorch 2.1.0, and CUDA 11.8.
Build-level and bitwise-identical reproduction across different hardware is not
guaranteed.

Compound and protein embeddings and ANNalog generation use separate Conda
environments because their dependency requirements conflict. These environments
are needed only when regenerating the corresponding intermediate data; they are
not required when using the published embeddings from Zenodo.

| Task | Environment file | Conda environment |
| --- | --- | --- |
| BaCNet training, inference, and ablation | `environments/bacnet.yml` | `bacnet` |
| ChemBERTa compound embeddings | `environments/chemberta.yml` | `chemberta_env` |
| Chemical Checker embeddings | `environments/chemical-checker.yml` | `cc_env` |
| ESM protein embeddings | `environments/esm.yml` | `esm_env` |
| Morgan fingerprints | `environments/morgan-fingerprint.yml` | `morgan_fingerprint_env` |
| ANNalog compound generation | `environments/annalog.yml` | `annalog` |

For example, to regenerate ESM embeddings:

```bash
conda env create -f environments/esm.yml
conda activate esm_env
```

See `environments/README.md` for the scope and platform requirements of each
environment.

⸻

Core dependencies

- Python 3.9.18
- 	PyTorch
- NumPy
- Pandas
- RDKit
- Transformers
- Signaturizer
- tqdm
- admet-ai

⸻

## Example Usage


### Generate chemical embeddings

#### Download the required Chemical Checker Signaturizer parameters

If you want to generate **ChemicalChecker (CC) / Signaturizer** embeddings locally, you need to download the CC model parameter archives first.

- You can run the download as a batch job if you are in an HPC environment.
- Runtime can vary depending on network and server load.

BaCNet uses ten Chemical Checker spaces: **A1–A5 and B1–B5**. The remaining
spaces are not required for BaCNet inference.

**1) Download the ten required model archives (A1–A5 and B1–B5)**

Run the following in any working directory:

```bash
# Download the ten spaces used by BaCNet
for letter in A B; do
  for num in {1..5}; do
    model="${letter}${num}"
    echo "Downloading ${model}..."
    wget "https://chemicalchecker.com/api/db/getSignaturizer/current/${model}" -O "${model}.tar.gz"
  done
done
```

**2) Verify the downloaded archives**

If you see some file listings, the archive is likely OK:

```bash
tar -tzf A1.tar.gz | head -n 20
```

**3) Extract archives into a parameter directory**

```bash
mkdir -p cc_param  # choose any directory name
for letter in A B; do
  for num in {1..5}; do
    model="${letter}${num}"
    mkdir -p "cc_param/${model}"
    tar -xzf "${model}.tar.gz" -C "cc_param/${model}"
  done
done
```

**4) (Optional) Inspect a CC model with TensorFlow Hub**

Chemical Checker models are provided in a TensorFlow format. You can sanity-check the extracted directory as follows:

```python
import tensorflow_hub as hub
local_dir = "path/to/cc_param/A1"
m = hub.load(local_dir)
print("signatures:", list(m.signatures.keys()))
```

> Note: You may need additional packages for the inspection step (e.g., `tensorflow` and `tensorflow_hub`).

Pass the directory containing the extracted `A1`–`A5` and `B1`–`B5`
subdirectories with `--cc-param-dir`. Do not edit the source code or place a
machine-specific absolute path in the repository.

```bash
python src/chemical_embedding.py \
    --input_csv examples/example_mols.csv \
    --cc-param-dir cc_param \
    --output-dir embeddings \
    --methods chemical_checker chemberta morgan \
    --admet_filter alert
```

The generated embeddings are saved under `embeddings/Chemical_embeddings`.
Compounds whose embedding generation failed are recorded in
`embeddings/embedding_failures.csv`, including the embedding type and error
message.

`chemical_embedding.py` CLI arguments

- `--input_csv` (required): Path to the input CSV containing compounds. The CSV must include **Compound_ID** and **SMILES** columns.
- `--cc-param-dir`: Directory containing the ten Chemical Checker model directories (`A1`–`A5` and `B1`–`B5`). Required only when `chemical_checker` is selected.
- `--output-dir` (optional): Output directory for filtered data, embeddings, and the failure report (default: `embeddings`).
- `--methods` (optional): One or more of `chemical_checker`, `chemberta`, and `morgan` (default: all three). This allows methods to run in separate package environments while retaining the same IDs and output contract.
- `--device` (optional): `auto`, `cpu`, or `cuda` for ChemBERTa (default: `auto`).
- `--chemberta-batch-size` (optional): ChemBERTa batch size (default: 16).
- `--resume` (optional): Reuse valid vectors already present in the selected output files.
- `--admet_ai_csv` (optional): Path to an ADMET-AI output CSV used for ADMET-based filtering.
- `--admet_filter` (optional): Filter compounds based on ADMET properties. Options: `none`, `flag`, or `alert` (default: `none`).

Chemical representations used by BaCNet are generated as follows:

- Morgan fingerprint: 1,024 dimensions (radius 2).
- Chemical Checker: ten 128-dimensional spaces concatenated in the order A1–A5 and B1–B5, for 1,280 dimensions.
- ChemBERTa: the 384-dimensional final-layer hidden state of the CLS token from `DeepChem/ChemBERTa-77M-MLM`, using `DeepChem/SmilesTokenizer_PubChem_1M`.

### Generate protein embeddings

Protein embeddings should be generated using **ESM-2**.

- Use the official ESM repository: https://github.com/facebookresearch/esm
- Model: `esm2_t48_15B_UR50D`
- Use **sequence representations** (named `sequence_representations` in the ESM repository) as the embedding.
  - Do **not** use token-level representations (`token_representation`).

Use the bundled wrapper to validate sequences, perform residue mean pooling, and
write a failure report and reproducibility manifest:

```bash
python src/esm_embedding.py \
    --input proteins.fasta \
    --output embeddings/protein_esm2.pt \
    --model-path /path/to/esm2_t48_15B_UR50D.pt \
    --device cuda \
    --batch-size 1 \
    --max-length 1022 \
    --long-sequence-policy error
```

CSV input uses `protein_id` and `sequence` columns. The previous `Name` and
`Sequence` column names are also accepted for compatibility. Sequence IDs must
be unique. The default maximum length is 1,022 residues. Long sequences are not
silently truncated; select `--long-sequence-policy truncate` only when that
scientific choice is intentional.

For the 15B model with CPU offloading, start the wrapper through `torchrun` so
the rendezvous settings are supplied by PyTorch instead of being fixed in the
source:

```bash
torchrun --standalone --nproc-per-node=1 src/esm_embedding.py \
    --input proteins.fasta \
    --output embeddings/protein_esm2.pt \
    --model-path /path/to/esm2_t48_15B_UR50D.pt \
    --backend fsdp \
    --device cuda
```

The protein embedding file is a serialized Python dictionary of the form
`{protein_id: 5120-dimensional CPU tensor}`.

An example is provided at `examples/target_protein/PBP_ecoli.pt`.

### Run BaCNet

```bash
python src/search_drug.py \
    --model models/checkpoint.pt \
    --ecdf models/ecdf_bacnet_v1.npz \
    --protein examples/target_protein/PBP_ecoli.pt \
    --chemical examples/chemical_library \
    --output outputs
```

`search_drug.py` CLI arguments

- `--model` (optional): Path to the trained model checkpoint (default: `models/checkpoint.pt`).
- `--ecdf` (optional): Path to the frozen ECDF reference file (default: `models/ecdf_bacnet_v1.npz`).
- `--protein` (required): Path to the target protein embedding file.
- `--chemical` (required): Base path to the chemical vector files.
- `--output` (optional): Directory to save per-protein screening CSV files (default: `outputs`; one file per protein, e.g. `{protein_name}_screening_score.csv`).

⸻

## Input Format

Example input CSV to generate chemical embeddings:

| Compound_ID | SMILES       |
| :---------- | :----------- |
| mol1 | CC(=O)O      |
| mol2 | C1=CC=CC=C1  |

Required columns:

- Compound_ID
- SMILES

⸻

## Output

The model outputs:

- `Compound_ID`: compound identifier (taken from the input `Compound_ID` field)
- `CPI_score`: predicted interaction score
- ranked compound list

The BaCNet input has 7,808 dimensions: ESM-2 (5,120), Morgan fingerprint
(1,024), Chemical Checker (1,280), and ChemBERTa CLS embedding (384). The
inference code validates every component and stops with an explicit error if a
dimension or value is invalid.

⸻

## Notes

- GPU is recommended for protein and chemical embedding generation
- Large chemical libraries may require significant memory.
- Ensure RDKit is correctly installed.

⸻

## Repository Structure

```text
bacnet/
├── src/                    # Core implementation
│   ├── chemical_embedding.py
│   ├── esm_embedding.py
│   ├── train.py
│   ├── train_ablation.py
│   ├── training.py
│   ├── training_data.py
│   ├── ablation_models.py
│   ├── aggregate_annalog_results.py
│   ├── search_drug.py
│   ├── model.py
│   ├── helper_functions.py
│   ├── sascorer.py
│   └── fpscores.pkl.gz
│
├── examples/               # Example inputs and reported-workflow manifests
│   ├── annalog_workflow/   # Figure 4 aggregation and traceability files
│   ├── ablation/           # Supplementary Table 6 reproduction protocol
│   └── training/           # Training data schema and Zenodo dataset link
├── configs/                # Training configuration examples
├── tests/                  # Data-contract and smoke tests
├── models/                 # Trained model checkpoints
├── environments/           # Version-pinned Conda environments by workflow stage
├── LICENSE
└── README.md
```

⸻
## Preparation of Training Data

The curated BaCNet interaction dataset and its fixed training, validation, and
test assignments are available from Zenodo:

- **DOI:** [10.5281/zenodo.23158020](https://doi.org/10.5281/zenodo.23158020)

1. Training data were constructed from compound-protein interaction pairs obtained from STITCH (version 5.0). Data corresponding to ESKAPEE bacteria were extracted on the basis of taxonomy IDs (Supplementary Table 13). Protein identifiers were then mapped to the corresponding proteins in STRING (version 10.0).
2. For each compound, canonical SMILES and desalted SMILES were generated using RDKit.
3. Redundant records with duplicated sequence-compound pairs were removed.
4. The combined score was subjected to Box-Cox transformation and subsequently normalized to a range of [0, 1] using min-max scaling, yielding the target values for model training.

The completed value from step 4 must be supplied in the `stitch_score` column.
The training program uses `stitch_score` directly: it does not apply Box-Cox
transformation, min-max scaling, clipping, thresholding, or any other target
conversion. Details of the transformation belong to the separate training-data
construction protocol.

## Train BaCNet

The version-pinned training environment is provided as
`environments/bacnet.yml`. The ESM wrapper is intentionally used from the
separate `environments/esm.yml` environment containing `fair-esm`; `fairscale`
is additionally needed only for `--backend fsdp`. This separation avoids the
dependency conflicts between BaCNet training and embedding generation.

Training links require the following columns:

- `protein_id`: key in the ESM-2 embedding dictionary
- `compound_id`: shared key in all chemical embedding dictionaries
- `stitch_score`: the combined_score obtained from STITCH, divided by 1,000
- `transformed_score`: the value derived from stitch_score using the Box-Cox transformation and min-max scaling applied to generate the training score
- `split`: fixed assignment of `train`, `validation`, or `test`

The dataset uses a **leave-one-protein-out** split. Unique protein groups are
divided before interaction rows are selected: 10% of protein groups are first
assigned to test, and then 10% of the remaining groups are assigned to
validation. Both operations use seed 123, giving approximately 81% train, 9%
validation, and 10% test by protein-group count. Interaction-row ratios can
differ because proteins have different numbers of linked compounds.

The supplied dataset is already split, so `train.py` never performs this
operation. The reference implementation is retained as
`reference_leave_one_protein_out_split()` in `src/training_data.py` for
methodological transparency. The loader also rejects a supplied dataset if the
same protein group appears across multiple splits. When different IDs represent
the same amino-acid sequence, provide `protein_group_id`; otherwise
`protein_id` is used as the grouping key.

The reference function preserves the first-appearance order returned by
`Series.unique()`, as in the original code. Exact regeneration therefore also
requires the same input row order and a compatible scikit-learn version. The
distributed links file remains the authoritative split assignment.

An empty schema example is provided at `examples/training/links_placeholder.csv`.
After downloading the dataset from Zenodo, configure its local paths in
`configs/train_example.yaml`, or override them on the command line:

```bash
accelerate launch src/train.py \
    --config configs/train_example.yaml \
    --links data/links.csv \
    --protein-embeddings data/protein_esm2.pt \
    --morgan-embeddings data/morgan_fingerprint.pt \
    --cc-embeddings data/chemical_checker.pt \
    --chemberta-embeddings data/chemberta-2.pt \
    --output-dir outputs/bacnet_training_01
```

Before training, BaCNet validates IDs, finite values, splits, and dimensions
(5,120 + 1,024 + 1,280 + 384 = 7,808). Validation errors are written to
`validation_report.csv`. Successful runs save the resolved configuration, input
checksums, split summary, epoch metrics, best and last resumable checkpoints,
an inference-compatible `model_best_state_dict.pt`, and held-out test
predictions. `dataset_row` in the prediction file points to the original links
CSV row.

Until the real links and embeddings are prepared, run the deterministic,
test-only CPU smoke workflow:

```bash
python src/train.py --config configs/train_smoke.yaml --synthetic
```

Synthetic data verify the software path only and must not be used for scientific
evaluation.

## Reproduce the architecture-ablation study

The baseline and all five alternative architectures from Supplementary Table 6
are implemented in `src/ablation_models.py`. The ablation program consumes the
published train, validation, and test files directly; it does not perform any
additional sampling or recreate the data split.

Each architecture is trained once using the fixed model seed `123`. The seed is
reset before every architecture so that all models use the same supplied data
and training order. Results are single fixed-seed runs, not averages over
multiple seeds. Minor numerical differences can occur across hardware and
software environments, and bitwise-identical results are not guaranteed.

See `examples/ablation/README.md` and `configs/ablation.yaml` for the data
contract, all architecture definitions, the complete command, and the reported
metrics.

## Reproduce the ANNalog expansion workflow

The Figure 4 analog-expansion workflow is documented in
`examples/annalog_workflow`. It records the ANNalog generation settings,
chemical filters, BaCNet scores, within-run ranks, and the three compounds
selected as parents for second-generation expansion.

Build the combined long-format table from the archived experiment directories:

```bash
python src/aggregate_annalog_results.py \
    --manifest examples/annalog_workflow/run_manifest.csv \
    --selection examples/annalog_workflow/selection_manifest.csv \
    --data-root /path/to/ANNalog-results \
    --output examples/annalog_workflow/annalog_candidates.csv \
    --summary-output examples/annalog_workflow/run_summary.csv
```

The reported workflow includes the first-generation medium/far runs and five
second-generation branches. The first-generation near run and the
`1st_medium_003` far branch were exploratory and are listed separately as
not included in the Figure 4 analysis. Structural predictions are treated as a
secondary prioritization aid, not as experimental evidence of binding.


## LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
