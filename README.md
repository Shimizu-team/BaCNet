# A Bacteria-Centric Deep Learning Framework for Prioritizing Antibacterial Compounds against Multidrug-Resistant Bacteria Using BaCNet


A computational pipeline for **compound–bacterial protein interaction (CPI) prediction** using
chemical embeddings and machine learning models.


This repository provides tools for:

- Generating chemical embeddings
- Running inference for CPI prediction
- Scoring compounds
- Searching and ranking candidate molecules
- Executing one ANNalog generation/filtering/BaCNet-ranking cycle and tracing
  the reported two-stage prioritization results

---

## Installation

### Core BaCNet environment

```bash
conda env create -f environments/bacnet.yml
conda activate bacnet
```

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

The Chemical Checker Signaturizer parameter archives used in this study were
downloaded from the `current` endpoint on 2025-08-13. The `current` endpoint is
retained below because the parameter archives could not be retrieved
successfully through the date-specific endpoint.

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
subdirectories with `--cc-param-dir`.

```bash
python src/chemical_embedding.py \
    --input_csv examples/example_mols.csv \
    --cc-param-dir cc_param \
    --output-dir embeddings \
    --methods chemical_checker chemberta morgan \
    --admet_filter alert
```

The generated embeddings are saved under `embeddings/Chemical_embeddings`.
The complete PAINS, SA, and QED decisions, including failure reasons, are saved
to `embeddings/filter_annotations.csv`.
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

CSV input uses `protein_id` and `sequence` columns. Sequence IDs must
be unique. The default maximum length is 1,022 residues. Long sequences are not
silently truncated; select `--long-sequence-policy truncate` only when that
scientific choice is intentional.

The protein embedding file is a serialized Python dictionary of the form
`{protein_id: 5120-dimensional CPU tensor}`.

An example is provided at `examples/target_protein/PBP_ecoli.pt`.

### Run BaCNet

Two species-held-out model checkpoints are provided:

- `models/checkpoint_ecoli.pt`: trained without *E. coli* K-12 CPIs and used for
  species-held-out inference on *E. coli* targets, including the PBP1A case study.
- `models/checkpoint_paeruginosa.pt`: trained without *P. aeruginosa* CPIs and
  used for species-held-out inference on *P. aeruginosa* targets.

The *E. coli*-held-out checkpoint is the default model used by the screening
CLI and the ANNalog workflow example.

```bash
python src/search_drug.py \
    --model models/checkpoint_ecoli.pt \
    --ecdf models/ecdf_bacnet_v1.npz \
    --protein examples/target_protein/PBP_ecoli.pt \
    --chemical examples/chemical_library \
    --output outputs
```

`search_drug.py` CLI arguments

- `--model` (optional): Path to the trained model checkpoint (default: `models/checkpoint_ecoli.pt`).
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

## Reported-result-to-resource map

All paths in this table are relative to the repository root. The table
distinguishes executable workflows from documentation-only resources. “Not
deposited” means that the current repository does not contain a dedicated
input, script, or expected-output artifact for that result.

| Reported result | Input data and identifiers | Script or implementation | Configuration and environment |
| --- | --- | --- | --- |
| Figure 1: BaCNet training and evaluation using the fixed protein-disjoint splits | Zenodo dataset ([10.5281/zenodo.23241408](https://doi.org/10.5281/zenodo.23241408)); schema and split contract in `examples/training/README.md` | `src/train.py`, `src/training.py`, `src/training_data.py`, and model definition in `src/model.py` | `configs/train_example.yaml`; `environments/bacnet.yml` |
| Supplementary Table 6: baseline and five architecture-ablation models | Zenodo ablation split files; expected filenames, row counts, and SHA-256 digests in `examples/ablation/README.md` and `examples/ablation/data_files.sha256` | `src/ablation_models.py`, `src/train_ablation.py`, `src/training.py`, and `src/training_data.py` | `configs/ablation.yaml`; `environments/bacnet.yml`; model seed 123 |
| Figures 2–3: BaCNet scoring of the mianserin–*E. coli* PBP1A case study | Mianserin SMILES in `examples/example_mols.csv`; PBP1A embedding in `examples/target_protein/PBP_ecoli.pt`; compound embeddings in `examples/chemical_library/` | `src/chemical_embedding.py`, `src/esm_embedding.py`, and `src/search_drug.py` | `models/checkpoint_ecoli.pt`; `models/ecdf_bacnet_v1.npz`; embedding environments in `environments/` |
| Figure 3, Figure 4c–e, and Supplementary Figures 9–10: Boltz-2 complex predictions | YAML inputs in `examples/structural_analysis/boltz2/inputs/figure3/` and `examples/structural_analysis/boltz2/inputs/figure4_figureS9_S10/` | `examples/structural_analysis/boltz2/run_boltz_predict.sh` | Boltz-2 2.1.1 settings and the figure-to-input map in `examples/structural_analysis/boltz2/README.md` |
| Figure 4: ANNalog expansion, chemical filtering, selection, and BaCNet-score aggregation | `examples/annalog_workflow/run_manifest.csv`, `selection_manifest.csv`, and the filter and score tables under `examples/annalog_workflow/published_runs/` | `src/run_annalog_bacnet.py` for first-generation medium/far generation and ranking; `src/aggregate_annalog_results.py` for the reported two-generation aggregation | `configs/annalog_bacnet_example.yaml`; separate ANNalog, embedding, and BaCNet environments in `environments/`; reported settings and seeds in `run_manifest.csv` |
| External 15-target benchmark | `examples/external_datasets/external_benchmark_targets.csv`; source dataset: Wong *et al.* (2022), [doi:10.15252/msb.202211081](https://doi.org/10.15252/msb.202211081) | No source-data transformation script; RpoB (`P0A8V2`) and RpoC (`P0A8T7`) are marked as excluded in the target list | Gene names, UniProt accessions, and inclusion status in `external_benchmark_targets.csv`; protein sequences can be retrieved using the listed UniProt accessions |
| BindingDB quantitative-affinity comparison | `BindingDB_All_202609_tsv.zip`, obtained separately; construction criteria in `examples/external_datasets/README.md` | `src/prepare_bindingdb.py` | `environments/bacnet.yml`; reference UniProt canonical sequences were retrieved on 2026-09-23 |
| DrugBank approved-compound screening library | DrugBank 5.1.12 XML obtained separately under an Academic License; construction criteria in `examples/external_datasets/README.md` | `src/prepare_drugbank.py` | `environments/bacnet.yml`; RDKit PAINS A/B/C catalogs |

The training and ablation programs calculate SHA-256 checksums for supplied
input files and store them in each run's `data_manifest.json`. The static
`examples/ablation/data_files.sha256` file records the authoritative digests of
the three ablation split files. The synthetic commands documented below are
software-path smoke tests only and do not reproduce the scientific results.

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
│   ├── run_annalog_generation.py
│   ├── prepare_annalog_compounds.py
│   ├── assemble_annalog_embeddings.py
│   ├── run_annalog_bacnet.py
│   ├── prepare_bindingdb.py
│   ├── prepare_drugbank.py
│   ├── search_drug.py
│   ├── model.py
│   ├── helper_functions.py
│   ├── sascorer.py
│   └── fpscores.pkl.gz
│
├── examples/               # Example inputs and reported-workflow manifests
│   ├── annalog_workflow/   # Figure 4 aggregation and traceability files
│   ├── annalog_demo/       # Fixed 30-compound medium-run scoring demo
│   ├── ablation/           # Supplementary Table 6 reproduction protocol
│   ├── external_datasets/  # BindingDB and DrugBank construction procedures
│   ├── structural_analysis/ # Boltz-2 inputs and documented prediction settings
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

- **DOI:** [10.5281/zenodo.23241408](https://doi.org/10.5281/zenodo.23241408)

### Download and place the deposited dataset

From the repository root, download `BaCNet_dataset.zip` from Zenodo and extract
it under `data/`:

```bash
mkdir -p data
curl -L "https://zenodo.org/records/23241408/files/BaCNet_dataset.zip?download=1" \
    -o data/BaCNet_dataset.zip
unzip data/BaCNet_dataset.zip -d data
```

The resulting paths should have the following structure:

```text
data/BaCNet_dataset/
├── training_dataset/
│   ├── all_data/
│   │   ├── protein_id_mapping.csv
│   │   ├── compound_id_mapping.csv
│   │   ├── all_links.csv
│   │   └── species_summary.csv
│   ├── training_dataset_ecoli_k12/
│   │   ├── train_links.csv
│   │   ├── validation_links.csv
│   │   ├── test_links.csv
│   │   └── links_ecol_k12.csv
│   └── training_dataset_paeruginosa/
│       ├── train_links.csv
│       ├── validation_links.csv
│       ├── test_links.csv
│       └── links_pseudomonas.csv
└── ablation_dataset/
    ├── protein_id_mapping.csv
    ├── compound_id_mapping.csv
    ├── train_links.csv
    ├── validation_links.csv
    └── test_links.csv
```

`training_dataset_ecoli_k12` contains the fixed train/validation/test data used
to train the *E. coli* K-12 species-held-out model: *E. coli* K-12 proteins are
excluded from these three training partitions, and `links_ecol_k12.csv` contains
the external species-held-out evaluation pairs. Likewise,
`training_dataset_paeruginosa` excludes *P. aeruginosa* proteins from its
train/validation/test partitions, and `links_pseudomonas.csv` contains the
corresponding species-held-out evaluation pairs.

The downloaded dataset and generated embeddings are ignored by Git and should
remain under `data/BaCNet_dataset/`; they must not be committed to the source-code
repository.

### Generate embeddings from the deposited mappings

The protein and compound mapping tables in `training_dataset/all_data/` contain
the unique amino-acid sequences and SMILES used by both species-held-out
datasets. Generate the 5,120-dimensional ESM-2 protein embeddings as follows:

```bash
conda env create -f environments/esm.yml

conda run --no-capture-output -n esm_env \
  python src/esm_embedding.py \
    --input data/BaCNet_dataset/training_dataset/all_data/protein_id_mapping.csv \
    --id-column ProteinID \
    --sequence-column sequence \
    --output data/BaCNet_dataset/embeddings/protein_esm2.pt \
    --model-path /path/to/esm2_t48_15B_UR50D.pt \
    --device cuda \
    --batch-size 1 \
    --max-length 1022 \
    --long-sequence-policy error
```

The compound mapping table uses `ChemID` and `SMILES`. The embedding script
accepts this deposited schema directly. Use `--skip-filters` because the
deposited mappings define the compounds in the fixed training datasets; PAINS,
SA, and QED screening must not be reapplied when regenerating their embeddings.
The three compound representations require separate version-locked environments:

```bash
conda env create -f environments/morgan-fingerprint.yml
conda env create -f environments/chemical-checker.yml
conda env create -f environments/chemberta.yml

# 1,024-dimensional Morgan fingerprints
conda run --no-capture-output -n morgan_fingerprint_env \
  python src/chemical_embedding.py \
    --input_csv data/BaCNet_dataset/training_dataset/all_data/compound_id_mapping.csv \
    --output-dir data/BaCNet_dataset/embeddings \
    --methods morgan \
    --skip-filters

# 1,280-dimensional Chemical Checker signatures
conda run --no-capture-output -n cc_env \
  python src/chemical_embedding.py \
    --input_csv data/BaCNet_dataset/training_dataset/all_data/compound_id_mapping.csv \
    --output-dir data/BaCNet_dataset/embeddings \
    --methods chemical_checker \
    --cc-param-dir /path/to/CC_param \
    --skip-filters

# 384-dimensional ChemBERTa embeddings
conda run --no-capture-output -n chemberta_env \
  python src/chemical_embedding.py \
    --input_csv data/BaCNet_dataset/training_dataset/all_data/compound_id_mapping.csv \
    --output-dir data/BaCNet_dataset/embeddings \
    --methods chemberta \
    --device cuda \
    --chemberta-batch-size 16 \
    --skip-filters
```

The generated files are written to
`data/BaCNet_dataset/embeddings/Chemical_embeddings/` as
`morgan_fingerprint.pt`, `chemical_checker.pt`, and `chemberta-2.pt`. The ESM-2
output is written to `data/BaCNet_dataset/embeddings/protein_esm2.pt`. Embedding
generation is computationally intensive; use `--resume` to continue an
interrupted compound-embedding run without discarding valid existing vectors.

1. Training data were constructed from compound-protein interaction pairs obtained from STITCH (version 5.0). Data corresponding to ESKAPEE bacteria were extracted on the basis of taxonomy IDs (Supplementary Table 21). Protein identifiers were then mapped to the corresponding proteins in STRING (version 10.0).
2. For each compound, canonical SMILES was generated using RDKit.
3. Redundant records with duplicated sequence-compound pairs were removed.
4. The STITCH `combined_score` was divided by 1,000 to obtain `stitch_score`.
5. `stitch_score` was subjected to Box-Cox transformation and subsequently normalized to a range of [0, 1] using min-max scaling, yielding `transformed_score`, the target used for model training.

The completed value from step 5 must be supplied in the `transformed_score`
column. The Zenodo training files provide this column. The training program uses
`transformed_score` directly: it does not apply Box-Cox
transformation, min-max scaling, clipping, thresholding, or any other target
conversion. Details of the transformation belong to the separate training-data
construction protocol.

## External Dataset Preparation

The reconstruction procedures for the *P. aeruginosa* BindingDB quantitative-
affinity datasets and the DrugBank 5.1.12 approved-compound library are
documented in `examples/external_datasets/README.md`. The source databases are
not redistributed. `src/prepare_bindingdb.py` and `src/prepare_drugbank.py`
reproduce the selection and canonicalization steps after the appropriately
licensed source files have been obtained.

## Train BaCNet

The version-pinned training environment is provided as
`environments/bacnet.yml`. The ESM wrapper is intentionally used from the
separate `environments/esm.yml` environment containing `fair-esm`; `fairscale`
is additionally needed only for `--backend fsdp`. This separation avoids the
dependency conflicts between BaCNet training and embedding generation.

When retained for provenance, `stitch_score` denotes the original STITCH
`combined_score` divided by 1,000. It is not read by the training program.

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

Before training, BaCNet trims surrounding whitespace from split labels and
normalizes them to lowercase. It validates IDs, duplicate protein-compound
pairs, the `[0, 1]` target range, splits, finite values, and dimensions
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

## Reproduce the architecture-ablation study

The baseline and all five alternative architectures from Supplementary Table 6
are implemented in `src/ablation_models.py`. The ablation program consumes the
published train, validation, and test files directly.

Each architecture is trained once using the fixed model seed `123`. The seed is
reset before every architecture so that all models use the same supplied data
and training order. Every architecture is trained for exactly 50 epochs without
early stopping. The checkpoint with the lowest validation loss across those 50
epochs is used for the held-out test evaluation.

See `examples/ablation/README.md` and `configs/ablation.yaml` for the data
contract, all architecture definitions, the complete command, and the reported
metrics.

## Execute one ANNalog expansion and ranking cycle

The executable wrapper covers one complete generation, filtering, embedding,
and BaCNet-ranking cycle. The reported two-stage Figure 4 analysis is traced in
`examples/annalog_workflow`, which records the ANNalog generation settings,
chemical filters, BaCNet scores, within-run ranks, and the three compounds
selected as parents for second-generation expansion.

### Run the first generation from Mianserin to BaCNet ranking

Install ANNalog 0.5 from the pinned upstream Git commit and verify its bundled
checkpoint and vocabulary as described in
`examples/annalog_workflow/README.md`. Then edit
`configs/annalog_bacnet_example.yaml` to specify that checkout's ANNalog
checkpoint and vocabulary and the Chemical Checker parameter directory. Run the
wrapper from an environment containing PyYAML, such as the BaCNet environment:

```bash
conda run --no-capture-output -n bacnet \
  python src/run_annalog_bacnet.py \
    --config configs/annalog_bacnet_example.yaml
```

The wrapper calls the separate Conda environments with `conda run` and performs
the following steps in sequence: seeded ANNalog medium/far generation from
Mianserin, canonicalization and deduplication, PAINS/SA/QED filtering, Morgan,
Chemical Checker, and ChemBERTa embedding, PBP1A BaCNet inference, and final
ranking. Use `--resume` to reuse completed stages or `--dry-run` to inspect the
commands. Outputs, failure tables, logs, checksums, and the resolved workflow
manifest are written below the configured `output_dir`.

This wrapper intentionally stops after first-generation ranking. Selection of
parents for the second generation used complementary complex-structure
assessment and is not automated.

### Run the fixed 30-compound scoring demonstration

`examples/annalog_demo` packages the first 30 filter-passing compounds from the
first-generation medium run and their three chemical embeddings. It can be
scored directly with the E. coli-held-out checkpoint:

```bash
conda run --no-capture-output -n bacnet \
  python src/search_drug.py \
    --model models/checkpoint_ecoli.pt \
    --ecdf models/ecdf_bacnet_v1.npz \
    --protein examples/target_protein/PBP_ecoli.pt \
    --chemical examples/annalog_demo/embeddings \
    --output outputs/annalog_demo
```

See `examples/annalog_demo/README.md` for the data contract, reference scores,
checksums, and the automated 20- and 30-compound equivalence checks. These demo
scores use the E. coli-held-out checkpoint and are not the Figure 4 scores.

Rebuild the combined long-format table for all seven reported runs from the
published filter and BaCNet-score tables:

```bash
python src/aggregate_annalog_results.py \
    --manifest examples/annalog_workflow/run_manifest.csv \
    --selection examples/annalog_workflow/selection_manifest.csv \
    --data-root examples/annalog_workflow \
    --output outputs/annalog_candidates.csv \
    --summary-output outputs/annalog_run_summary.csv
```

The reported workflow includes the first-generation medium/far runs and five
second-generation branches. The first-generation near run and the
`1st_medium_003` far branch were exploratory and are listed separately as
not included in the Figure 4 analysis. Structural predictions are treated as a
secondary prioritization aid, not as experimental evidence of binding.

## Reproduce the Boltz-2 complex predictions

The Boltz input YAML files used for the currently collected Figure 3, Figure 4,
Supplementary Figure 9, and Supplementary Figure 10 complexes are provided in
`examples/structural_analysis/boltz2`.

See `examples/structural_analysis/boltz2/README.md` for the figure-to-input map,
the Boltz-2 version and exact command-line settings, and how the MSA generated
by the Boltz-2 MSA server was reused to reduce computation.


## LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
