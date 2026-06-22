# A Bacteria-Centric Deep Learning Framework for Antibacterial Hit Priorization against Multidrug-Resistant Bacteria Using BaCNet


A computational pipeline for **compound–bacterial protein interaction (CPI) prediction** using
chemical embeddings and machine learning models.

![alt text](<Graphical abstract.png>)

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

#### (Optional) Download ChemicalChecker Signaturizer parameters

If you want to generate **ChemicalChecker (CC) / Signaturizer** embeddings locally, you need to download the CC model parameter archives first.

- You can run the download as a batch job if you are on an HPC environment.
- Runtime can vary depending on network and server load.

**1) Download all model archives (A1–E5)**

Run the following in any working directory:

```bash
# Download all combinations: A–E and 1–5
for letter in {A..E}; do
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
for letter in {A..E}; do
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

After that, you can specify this directory in the CC_PARAM_DIR variable inside chemical_embedding.py to generate ChemicalChecker Signaturizer embeddings.

```bash
python src/chemical_embedding.py \
    --input examples/example_mols.csv \
    --output outputs/embeddings.pt
```

`chemical_embedding.py` CLI arguments

- `--input_csv` (required): Path to the input CSV containing compounds. The CSV must include **Name** and **SMILES** columns.

### Generate protein embeddings

Protein embeddings should be generated using **ESM-2**.

- Use the official ESM repository: https://github.com/facebookresearch/esm
- Model: `esm2_t48_15B_UR50D`
- Use **sequence representations** (named `sequence_representations` in the ESM repository) as the embedding.
  - Do **not** use token-level representations (`token_representation`).

The protein embedding file must be a serialized python dictionary of the form:
`{protein_name: embedding}`

An example is provided at `examples/target_protein/PBP_ecoli.pt`.

### Run BaCNet

```bash
python src/search_drug.py --model models/checkpoint.pt \
    --protein examples/target_protein/PBP_ecoli.pt\
    --chemical examples/chemical_library \
    --output outputs \
```

`search_drug.py` CLI arguments

- `--model` (optional): Path to the trained model checkpoint (default: `../model/checkpoint.pt`).
- `--protein` (required): Path to the target protein embedding file.
- `--chemical` (required): Base path to the chemical vector files.
- `--output` (optional): Path to save inference results for each protein (default: `{protein_name}_screening_score.csv`).

⸻

## Input Format

Example input CSV to generate chemical embeddings:

| Name | SMILES       |
| :--- | :----------- |
| mol1 | CC(=O)O      |
| mol2 | C1=CC=CC=C1  |

Required columns:

- Name
- SMILES

⸻

## Output

The model outputs:

- Name: Compound name
- CPI_score: Predicted interaction scores
- Ranked compound list

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
│   ├── search_drug.py
│   ├── model.py
│   ├── helper_functions.py
│   ├── sascorer.py
│   └── fpscores.pkl.gz
│
├── examples/               # Example input files
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


## LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
