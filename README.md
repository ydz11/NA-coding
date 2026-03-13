# cos-reco Repro Steps (Multi-factor NDCG@5/10/20)

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Prepare data

Put `ml-1m.zip` at project root:

`C:\Users\yaozh\Downloads\task\cos-reco\ml-1m.zip`

Or keep extracted data at:

`C:\Users\yaozh\Downloads\task\cos-reco\data\ml-1m`

## 3) Run full multi-factor validation

Default config already uses:

- factors: `8,16,32,64`
- seeds per factor: `42,43,44`
- train epochs: `5`
- report ks: `5,10,20`

Run:

```powershell
.\.venv\Scripts\python.exe main.py --stage sweep
```

## 4) Check outputs

Main comparison plots:

`outputs\ndcg_compare.png`
`outputs\ndcg_compare_k5.png`
`outputs\ndcg_compare_k10.png`
`outputs\ndcg_compare_k20.png`

Main aggregated metrics:

`outputs\results.json`

You can also run eval only (reuse checkpoints):

```powershell
.\.venv\Scripts\python.exe main.py --stage eval
```

Per-factor summaries:

- `outputs\factor_8\results.json`
- `outputs\factor_16\results.json`
- `outputs\factor_32\results.json`
- `outputs\factor_64\results.json`
