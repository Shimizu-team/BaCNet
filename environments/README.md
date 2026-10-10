# Conda environments

BaCNet and its data-preparation stages use separate Conda environments because
the required versions of Python, PyTorch, TensorFlow, CUDA, RDKit, and related
packages are not mutually compatible. Only `bacnet.yml` is required for BaCNet
training, inference, and the architecture-ablation study.

The other YAML files are required only when regenerating the corresponding
embeddings or ANNalog compounds.

`bacnet.yml` does not include RDKit. Use `chemical_filters.yml` for standalone
RDKit-dependent dataset preparation, including the BindingDB and DrugBank
scripts. The executable ANNalog workflow uses the RDKit installation in the
Morgan fingerprint environment for compound preparation and filtering.

| File | Purpose | Environment name | Recorded Python / accelerator stack |
| --- | --- | --- | --- |
| `bacnet.yml` | BaCNet training, inference, and ablation | `bacnet` | Python 3.9.18, PyTorch 2.1.0, CUDA 11.8 |
| `chemberta.yml` | ChemBERTa compound embeddings | `chemberta_env` | Python 3.9.18, PyTorch 2.1.0, CUDA 11.8 |
| `chemical-checker.yml` | Chemical Checker embeddings | `cc_env` | Python 3.7.12, PyTorch 1.9.0 (CPU), TensorFlow 2.11.0, Signaturizer 1.1.14 |
| `esm.yml` | ESM protein embeddings | `esm_env` | Python 3.9.17, PyTorch 1.11.0, CUDA Toolkit 11.3 |
| `morgan-fingerprint.yml` | Morgan fingerprints | `morgan_fingerprint_env` | Python 3.8.19, RDKit 2024.03.2 |
| `annalog.yml` | ANNalog compound generation | `annalog` | Python 3.9.2, PyTorch 1.9.1 |
| `chemical_filters.yml` | SMILES canonicalization, PAINS, SA score, and QED filtering | `chemical_filters` | Python 3.12.3, RDKit 2024.03.5 |

Create the required environment from the repository root. For example:

```bash
conda env create -f environments/bacnet.yml
conda activate bacnet
```

The ANNalog environment requires one post-creation installation step because
the reported ANNalog 0.5 code was installed from a pinned upstream Git commit,
not from PyPI. Follow the clone, checkout, installation, and checkpoint/vocabulary
verification instructions in `examples/annalog_workflow/README.md`.

The automated tests likewise span two environments. Run the core tests in
`bacnet` and the RDKit-dependent tests in `morgan_fingerprint_env`; the exact
commands and test-module assignments are documented in the main `README.md`.

The Signaturizer 1.1.14 distribution retains an internal `__version__` value of
`1.1.13`. Use package metadata (for example, `python -m pip show signaturizer`)
when verifying the installed Signaturizer release.
