# Conda environments

BaCNet and its data-preparation stages use separate Conda environments because
the required versions of Python, PyTorch, TensorFlow, CUDA, RDKit, and related
packages are not mutually compatible. Only `bacnet.yml` is required for BaCNet
training, inference, and the architecture-ablation study.

The other YAML files are required only when regenerating the corresponding
embeddings or ANNalog compounds.

| File | Purpose | Environment name | Recorded Python / accelerator stack |
| --- | --- | --- | --- |
| `bacnet.yml` | BaCNet training, inference, and ablation | `bacnet` | Python 3.9.18, PyTorch 2.1.0, CUDA 11.8 |
| `chemberta.yml` | ChemBERTa compound embeddings | `chemberta_env` | Python 3.9.18, PyTorch 2.1.0, CUDA 11.8 |
| `chemical-checker.yml` | Chemical Checker embeddings | `cc_env` | Python 3.7.12, TensorFlow 2.11.0 |
| `esm.yml` | ESM protein embeddings | `esm_env` | Python 3.9.17, PyTorch 1.11.0, CUDA Toolkit 11.3 |
| `morgan-fingerprint.yml` | Morgan fingerprints | `morgan_fingerprint_env` | Python 3.8.19, RDKit 2024.03.2 |
| `annalog.yml` | ANNalog compound generation | `annalog` | Python 3.9.25, PyTorch 2.5.1, CUDA 12.1 |
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
