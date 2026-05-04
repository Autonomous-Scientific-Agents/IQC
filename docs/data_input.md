# Tabular Data Input

IQC supports tabular input files through `--input` / `-i`. This mode is useful when molecule structures are stored in a data table rather than in separate `.xyz` files.

## Inspecting Data

Run `iqc --input FILE` with no other options to inspect a data file:

```bash
iqc --input molecules.parquet
```

The inspection report includes:

- file format and reader engine
- file size
- row and column counts
- column names and types
- null counts and null percentages
- min, max, and mean for numeric columns

## Supported Formats

The tabular reader supports:

- parquet: `.parquet`, `.pq`
- delimited text: `.csv`, `.tsv`, `.tab`, `.txt`
- Excel: `.xls`, `.xlsx`, `.xlsm`, `.ods`
- JSON: `.json`, `.jsonl`, `.ndjson`
- Feather: `.feather`, `.ftr`
- Arrow IPC: `.arrow`, `.ipc`

For parquet, CSV/TSV, Feather, and Arrow IPC files, IQC reads only the requested structure column where the backend supports column projection. Excel and some JSON layouts are read through pandas.

## Running Calculations From Data Columns

When `--input` is used with a calculation task, pass exactly one structure column selector.

Use `--xyz COLUMN` when the data file contains XYZ-format geometry text:

```bash
iqc --input molecules.parquet --xyz geometry --task single
```

Use `--smiles COLUMN` when the data file contains SMILES strings:

```bash
iqc --input molecules.csv --smiles smiles --task opt
```

The normal calculation options still apply:

```bash
iqc --input molecules.parquet \
  --smiles smiles \
  --task nmr \
  --backend orca \
  --output-dir nmr_results
```

`--xyz` and `--smiles` are mutually exclusive when `--input` is provided. Passing neither selector with calculation options is also an error because IQC cannot infer which column contains the structure.

## XYZ Columns

XYZ columns should contain a complete XYZ block per row:

```text
3
water
O 0 0 0
H 0 0 1
H 1 0 0
```

CSV files can store multiline XYZ cells, but the cell must be quoted:

```csv
name,geometry
water,"3
water
O 0 0 0
H 0 0 1
H 1 0 0
"
```

Blank or missing XYZ cells are rejected with the row index in the error message.

## SMILES Columns

SMILES columns should contain one SMILES string per row:

```csv
name,smiles
water,O
ethanol,CCO
```

IQC strips leading and trailing whitespace from SMILES values, then uses the existing RDKit path to generate a 3D geometry. Blank or missing SMILES cells are rejected with the row index in the error message.

## Result Metadata

Results from tabular inputs include source metadata:

- `input_mode`: `data_xyz` or `data_smiles`
- `data_input_file`: source data file path
- `data_xyz_column`: XYZ column name when using `--xyz`
- `data_smiles_column`: SMILES column name when using `--smiles`
- `data_row_index`: zero-based row index from the source file

This metadata lets downstream workflows trace each calculation result back to the original data row.
