# A Bacteria-Centric Deep Learning Framework for Antibacterial Hit Prioritization against Multidrug-Resistant Bacteria Using BaCNet


A computational pipeline for **compound–bacterial protein interaction (CPI) prediction** using
chemical embeddings and machine learning models.


This repository provides tools for:

- Generating chemical embeddings
- Running inference for CPI prediction
- Scoring compounds
- Searching and ranking candidate molecules

---

## Installation

### Option 1: Conda (Recommended)

1) Create and activate an environment

```bash
conda create -n bacnet python=3.11 -y
conda activate bacnet

conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
```

2) Install PyTorch

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3) Install dependencies and other packages

```bash
conda install -y rdkit pandas numpy matplotlib reportlab pyarrow tqdm transformers
conda install -y "setuptools<81"
python -m pip install signaturizer
```

---

### Option 2: Using provided script

```bash
conda env create -f environment.yaml
conda activate bacnet
```

Note: You may need to edit the script depending on your environment.

⸻

Dependencies

- Python >= 3.10
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
│   ├── training.py
│   ├── training_data.py
│   ├── search_drug.py
│   ├── model.py
│   ├── helper_functions.py
│   ├── sascorer.py
│   └── fpscores.pkl.gz
│
├── examples/               # Example input files
├── configs/                # Training configuration examples
├── tests/                  # Data-contract and smoke tests
├── models/                 # Trained model checkpoints
├── LICENSE
├── environment.yaml
└── README.md
```

⸻
## Preparation of Training Data
1. Training data were constructed from compound-protein interaction pairs obtained from STITCH (version 5.0). Data corresponding to ESKAPEE bacteria were extracted on the basis of taxonomy IDs (Supplementary Table 13). Protein identifiers were then mapped to the corresponding proteins in STRING (version 10.0).
2. For each compound, canonical SMILES and desalted SMILES were generated using RDKit.
3. Redundant records with duplicated sequence-compound pairs were removed.
4. The combined score was subjected to Box-Cox transformation and subsequently normalized to a range of [0, 1] using min-max scaling, yielding the target values for model training.

The completed value from step 4 must be supplied in the `target_bct` column.
The training program uses `target_bct` directly: it does not apply Box-Cox
transformation, min-max scaling, clipping, thresholding, or any other target
conversion. Details of the transformation belong to the separate training-data
construction protocol.

## Train BaCNet

The training environment requires `accelerate` and `pyyaml` in addition to
PyTorch, NumPy, and Pandas. The ESM wrapper is intentionally usable from a
separate environment containing `fair-esm`; `fairscale` is additionally needed
only for `--backend fsdp`. This separation allows embedding environments to be
managed independently until reproducible environment files are added.

Training links require the following columns:

- `protein_id`: key in the ESM-2 embedding dictionary
- `compound_id`: shared key in all chemical embedding dictionaries
- `target_bct`: finite, already transformed regression target
- `split`: fixed assignment of `train`, `validation`, or `test`

An empty schema placeholder is provided at
`examples/training/links_placeholder.csv`. Configure real paths in
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


## LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
