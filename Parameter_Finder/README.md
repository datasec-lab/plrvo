# PLRV-O Parameter Finder

## Files

```text
parameter_finder/
├── param_finder.py
├── find_plrvo.py
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
python param_finder.py
```

## Configuration

Set the following parameters in `param_finder.py`:

- `C` — clipping value(s).
- `MODEL_SIZE` — number of model parameters.
- `DATASET_SIZE` — number of training samples.
- `BATCH` — batch size.
- `EPOCHS` — training epochs.
- `EPSILON_TARGET` — target privacy budget.
- `DELTA` — privacy delta.

Run the script 

## Output

```text
training_candidates.csv
```
