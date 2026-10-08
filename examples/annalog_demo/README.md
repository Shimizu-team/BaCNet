# Fixed 30-compound ANNalog/BaCNet demonstration

This compact example contains the first 30 filter-passing compounds, in source
order, from the first-generation ANNalog **medium** run. It avoids identifier
collisions with other ANNalog branches and keeps the repository footprint
small while exercising BaCNet scoring with real generated compounds and
precomputed embeddings.

## Included files

- `compounds.csv`: compound identifiers, SMILES, ANNalog score, similarities,
  and PAINS/SA/QED filter values.
- `embeddings/`: Morgan (1,024), Chemical Checker (1,280), and ChemBERTa-2
  (384) vectors for the same 30 identifiers. All stored values are one-dimensional
  `float32` PyTorch tensors.
- `expected/P02918_screening_score.csv`: reference BaCNet scores for PBP1A
  (UniProt P02918).
- `demo_manifest.json`: selection rule, identifiers, dimensions, and SHA-256
  checksums for source, reference, and packaged files.

The expected scores were extracted from a run that scored the complete
694-compound medium library with `models/checkpoint_ecoli.pt`.

## Run the demonstration

From the repository root, in the BaCNet environment:

```bash
conda run --no-capture-output -n bacnet \
  python src/search_drug.py \
    --model models/checkpoint_ecoli.pt \
    --ecdf models/ecdf_bacnet_v1.npz \
    --protein examples/target_protein/PBP_ecoli.pt \
    --chemical examples/annalog_demo/embeddings \
    --output outputs/annalog_demo
```

The resulting `outputs/annalog_demo/P02918_screening_score.csv` should match the
packaged expected scores (floating-point tolerance `1e-8`). The automated test
checks both the first 20 compounds requested for validation and all 30 packaged
compounds:

```bash
python -m unittest tests.test_annalog_demo
```
