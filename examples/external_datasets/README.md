# External dataset preparation

This directory documents the construction of the BindingDB quantitative-
affinity datasets and the approved-compound DrugBank screening library. The
source databases are not redistributed in this repository. Users must obtain
them under the terms of their respective providers and run the preparation
scripts locally.

The scripts only canonicalize the supplied SMILES with RDKit. They do **not**
desalt, neutralize, or perform tautomer standardization.

## BindingDB

### Source data

Use `BindingDB_All_202609_tsv.zip` as the input file. The analysis uses only
records assigned to *Pseudomonas aeruginosa* by scientific name.

### Record selection

Ki, Kd, and IC50 are handled as three separate datasets because they represent
different measurements. A record is retained when all of the following
conditions are satisfied:

1. the target organism field contains *Pseudomonas aeruginosa*;
2. the Ki, Kd, or IC50 value is recorded in the corresponding nM column;
3. the value is positive and finite, and its relation is exactly `=`;
4. the target contains one protein chain;
5. a UniProt accession and a BindingDB target sequence are available;
6. the BindingDB sequence exactly matches the current UniProt canonical
   sequence;
7. the target name does not indicate a mutant, variant, engineered protein,
   partial sequence, fragment, construct, or truncated protein; and
8. the canonical protein sequence contains no more than 1,022 residues, which
   is the ESM-2 input limit used in this study.

Ligand SMILES are parsed and converted to isomeric canonical SMILES with
RDKit. Structures that RDKit cannot parse are excluded. No other chemical
structure standardization is performed.

The measured value is converted to pAffinity as

```text
pAffinity = 9 - log10(affinity in nM)
```

Replicate measurements for the same UniProt accession, canonical SMILES, and
endpoint are represented by the median pAffinity. Ki, Kd, and IC50 values are
never pooled with one another.

### Reproduction

The script queries the UniProt REST API to retrieve current canonical
sequences. Responses are cached in the output directory so that an interrupted
run can resume without repeating completed queries.

```bash
python src/prepare_bindingdb.py \
  --input /path/to/BindingDB_All_202609_tsv.zip \
  --output-dir outputs/bindingdb
```

The command writes separate
`pseudomonas_aeruginosa_ki.csv`,
`pseudomonas_aeruginosa_kd.csv`, and
`pseudomonas_aeruginosa_ic50.csv` files.

## DrugBank

### Source and license

DrugBank version 5.1.12 was obtained under an Academic License. DrugBank data
are not included in this repository. Each user must obtain an appropriately
licensed DrugBank 5.1.12 XML file directly from DrugBank.

### Approved-compound selection and PAINS filtering

The screening library is constructed as follows:

1. retain records whose DrugBank `groups` field contains `approved`;
2. remove records without a SMILES value;
3. convert the supplied SMILES to an isomeric canonical SMILES with RDKit;
4. remove structures that RDKit cannot parse; and
5. apply all RDKit PAINS catalogs (PAINS A, PAINS B, and PAINS C) and remove
   every compound matching at least one catalog.

Canonicalization is the only structure preprocessing step. The script does not
desalt or neutralize compounds.

### Reproduction

The script accepts either a plain DrugBank XML file or a ZIP archive containing
one XML file.

```bash
python src/prepare_drugbank.py \
  --input /path/to/drugbank_5.1.12.xml.zip \
  --output outputs/drugbank_approved.csv
```

The resulting CSV contains the retained DrugBank identifier as `Compound_ID`
and its canonical structure as `SMILES`, matching the BaCNet compound-input
convention.
