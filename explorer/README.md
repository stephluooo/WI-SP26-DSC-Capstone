# DDI Explorer

An interactive Streamlit dashboard for exploring the Plan C drug–drug interaction
study results in `results/ddi_study/`.

## Quick start

```bash
pip install -r explorer/requirements.txt
python explorer/build_db.py          # one-time, ~1.5 min (already done locally)
python -m streamlit run explorer/app.py
```

Opens at `http://localhost:8501`. (Use `python -m streamlit` — the `streamlit`
script isn't on PATH on this machine.)

## What's inside

| Page | Feature |
|------|---------|
| Overview | Headline stats + pipeline figures |
| Drug Pair | All reactions for a pair, ROR table, **2×2 contingency visualizer** |
| Drug Profile | A drug's top partner drugs and reactions |
| Reaction | Which pairs signal for a given reaction |
| Leaderboard | Global sortable/filterable signal table (absolute ROR vs robust CI lower bound) |
| Network | Ego interaction network around a focus drug |
| Heatmap | Partner × reaction signal-intensity heatmap |
| Volcano | ROR vs evidence scatter + ROR distribution |
| Predictions | Top-500 novel predictions, flagged against known DrugBank DDIs |
| Model | Cross-validation metrics, ROC, Precision@k |
| Similarity | Tanimoto fingerprint similarity + **live model scoring** of any pair |
| Audit | FAERS → DrugBank canonicalization audit |

Every table has a **Download CSV** button (autocomplete search is built into the pickers).

## Setup

From the **project root** (`c:\Users\mkbox\Documents\Cursor`):

```bash
pip install -r explorer/requirements.txt
```

## 1. Build the database (one-time, ~1–2 min)

This streams the 650 MB signals CSV into an indexed SQLite file
(`results/ddi_study/ddi.db`):

```bash
python explorer/build_db.py
```

Quick rebuild of just the small tables (skips the big signals load):

```bash
python explorer/build_db.py --skip-signals
```

## 2. Run the app

```bash
python -m streamlit run explorer/app.py
```

It opens at `http://localhost:8501`. (Use `python -m streamlit` because the
`streamlit` script isn't on PATH on this machine.)

## Optional data

- **Known DrugBank DDIs** (`data/drugbank_all_drug_drug_interactions.csv`): if present,
  the Predictions and Drug Pair pages flag whether a pair is already documented.
- **Fingerprints** (`phase2_fingerprints.npz`) and **model** (`best_model.pt`): required
  for the Similarity page (Tanimoto + live scoring). The rest of the app works without them.

## Notes / hosting

- `ddi.db` is generated and large — it is git-ignored. Rebuild it with `build_db.py`
  on any machine that has the result CSVs.
- This is a **local** app. To host it publicly you'd run the same command on a server
  (e.g. an EC2/Lightsail instance or Streamlit Community Cloud) with the result files
  and `ddi.db` present, and expose port 8501 behind a reverse proxy.
