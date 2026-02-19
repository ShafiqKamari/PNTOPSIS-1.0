# PNTOPSIS Streamlit App (Pythagorean Neutrosophic TOPSIS)

This repository provides a Streamlit implementation of **PNTOPSIS**, where a crisp decision matrix is mapped to **Pythagorean Neutrosophic Set (PNS)** triplets and ranked using TOPSIS with a **normalized PN–Euclidean distance**.

## Features
- Crisp scores → PNS triplets using **strict lookup** tables (5 / 7 / 9 / 11-point scales).
- Criterion types: **Benefit (B)** / **Cost (C)**:
  - Read from file if the first row is `B/C`, or
  - Set/edit in the UI before computation.
- Weights:
  - **Equal weights** button, or
  - **Manual weights** input (**no auto-normalization**).
  - App shows a **warning** if \(\sum_j w_j \neq 1\) but still proceeds.
- PIS/NIS definitions:
  - Benefit: \(V^+ = (\max \tau, \min \xi, \min \eta)\), \(V^- = (\min \tau, \max \xi, \max \eta)\)
  - Cost:    \(V^+ = (\min \tau, \max \xi, \max \eta)\), \(V^- = (\max \tau, \min \xi, \min \eta)\)
- Distance:
  - \(d = \sqrt{\frac{1}{3n} \sum_{j=1}^{n} [ (\tau-\tau^*)^2 + (\xi-\xi^*)^2 + (\eta-\eta^*)^2 ] }\)
- Exports all intermediate matrices and results to Excel.

## File format (CSV / Excel)

### Option A: With criterion types row (recommended)
- **Row 1** (top row across criteria): criterion types: `B` or `C`
- **Rows 2..**: crisp decision matrix scores

You may optionally include a first column of alternative names.

Example:

| Alt | C1 | C2 | C3 |
|---|---|---|---|
|   | B | C | B |
| A1 | 7 | 3 | 5 |
| A2 | 6 | 2 | 4 |

### Option B: Without criterion types row
Upload only the crisp matrix. The app will default all criteria to **Benefit (B)** and you can adjust in the UI before running.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (root contains `app.py` and `requirements.txt`)
2. In Streamlit Cloud: **New app** → select repo → set main file path to `app.py`
3. Deploy

## Validation rules
- Scores must be integers in the selected scale range:
  - 5-point: 1..5
  - 7-point: 1..7
  - 9-point: 1..9
  - 11-point: 1..11
- Out-of-range values cause **error + stop**.
