# ML Platform

A full-stack data science web app. Upload a CSV, clean it, profile it, train ML models, and export — all in one browser tab.

**Stack:** FastAPI · React + Vite + TypeScript · scikit-learn · pandas · matplotlib · ydata-profiling

---

## What it does

| Tab | Purpose |
|---|---|
| **Upload** | Upload a CSV, preview first 50 rows, optionally promote first row to column headers |
| **Clean** | Change types, rename columns, fill/drop missing values, remove duplicates, one-hot / label encode |
| **Analyze** | Pearson correlation matrix heatmap + full ydata-profiling HTML report |
| **Model** | Train Linear Regression, Decision Tree, or KNN with hyperparameter dropdowns — see metrics + plots |
| **Export** | Download the cleaned dataset as CSV |

---

## Project structure

```
ml-platform/
├── backend/
│   ├── main.py          # FastAPI app — all endpoints in one file
│   ├── requirements.txt
│   └── reports/         # Saved profile report HTML files
└── frontend/
    ├── src/
    │   ├── api/client.ts        # All fetch calls
    │   ├── types/api.ts         # TypeScript interfaces
    │   ├── components/          # NavBar, PreviewTable, HyperparamPanel
    │   ├── tabs/                # UploadTab, CleanTab, AnalyzeTab, ModelTab, ExportTab
    │   └── styles/              # Plain CSS per component
    └── .env                     # VITE_API_URL=http://localhost:8000
```

---

## Live Deployment

🚀 **Try the deployed app:** [https://data-science-platform-six.vercel.app/](https://data-science-platform-six.vercel.app/)

- **Frontend:** Deployed on Vercel
- **Backend:** Deployed on Render (free tier)

⚠️ **Important:** The backend spins down after 15 minutes of inactivity. The first request after inactivity may take up to 1 minute to respond while the server restarts (cold start).

> **For local development instructions**, see `STARTING_PROJECT.txt` in the root directory.

---

## ML endpoints

```
POST /api/model/linear_regression   fit_intercept, test_size
POST /api/model/decision_tree       max_depth, min_samples_split, criterion, test_size
POST /api/model/knn                 n_neighbors, weights, metric, test_size
```

Each returns `metrics`, optional `viz_base64` (2D plot when ≤ 2 features), and `tree_base64` (Decision Tree only).

---

## Notes

- Text columns (object dtype) must be encoded in the **Clean tab** before using as model features.
- Profile report generation can take 30–60 s on large datasets.
- One dataset in memory at a time. Uploading a new file replaces the current one.
