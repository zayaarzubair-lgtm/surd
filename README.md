# SURDview

A dashboard for exploring **SURD-style information decomposition** — see how much each source variable uniquely, redundantly, or synergistically predicts a target.

> **Week 1 MVP:** the SURD numbers are deterministic placeholders. Swap in the real algorithm later by editing only `src/components/surd_engine.py`.

---

## Quick start (Windows PowerShell)

```powershell
# 1. Clone or download this folder, then cd into it
cd surdview

# 2. Create a virtual environment
python -m venv .venv

# 3. Allow scripts to run (current session only)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Activate
.\.venv\Scripts\Activate.ps1

# 5. Install dependencies
pip install -r requirements.txt

# 6. Launch the dashboard
streamlit run app.py
```

### macOS / Linux alternative

```bash
cd surdview
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage

1. Upload any CSV with numeric columns (or tick **Use example weather dataset**).
2. Pick a **target (Y)** column and at least **2 source (X)** columns.
3. Adjust bins and missing-value strategy if needed.
4. Click **Run Analysis**.
5. Explore the four tabs: Overview chart, Details heatmap, Explanation, and Export.

---

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `dash_app.py` | Dash placeholder (future) |
| `src/pipeline/analysis_pipeline.py` | Single entry point the UI calls |
| `src/components/` | Individual steps: ingest, transform, SURD, explain |
| `src/utils.py` | Plotly chart builders |
| `data/example_small.csv` | Built-in demo dataset |
| `docs/design.md` | Architecture notes |

---

## Replacing the dummy SURD engine

Edit `src/components/surd_engine.py` and rewrite `compute_surd()`. It receives a discretised DataFrame plus column names and must return a dict with keys `unique`, `redundant`, `synergy`, and `pairwise_synergy`. Nothing else in the project needs to change.

---

## Requirements

- Python 3.10+
- streamlit, pandas, numpy, plotly (see `requirements.txt`)
