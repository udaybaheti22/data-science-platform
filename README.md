# ML Platform

A full-stack data science web app. Upload a CSV, clean it, profile it, get AI-powered suggestions, train ML models, and export — all in one browser tab.

**Stack:** FastAPI · React + Vite + TypeScript · scikit-learn · pandas · matplotlib · ydata-profiling · Gemini API

---

## What it does

| Tab | Purpose |
|---|---|
| **Upload** | Upload a CSV, preview first 50 rows, optionally promote first row to column headers |
| **Clean** | Change types, rename columns, fill/drop missing values, remove duplicates, one-hot / label encode |
| **Analyze** | Pearson correlation matrix heatmap + full ydata-profiling HTML report |
| **Model** | Train Linear Regression, Decision Tree, or KNN with hyperparameter dropdowns — see metrics + plots |
| **AI Suggestions** | Select a target column and task type — Gemini AI analyzes the dataset and recommends cleaning operations and optimal model hyperparameters |
| **Export** | Download the cleaned dataset as CSV |

---

## AI Suggestions feature

The **AI Suggestions** tab uses Google Gemini to analyze your dataset and return structured recommendations:

- **Cleaning suggestions** — per-column actions (fill missing, encode, drop, etc.) with reasons, shown as inline hints in the Clean tab
- **Model recommendation** — which of the 3 models to use and suggested hyperparameter values, shown as hints in the Model tab
- Results persist across tab navigation — no need to regenerate after switching tabs

Requires a `GEMINI_API_KEY` environment variable on the backend (see setup instructions in `STARTING_PROJECT.txt`).

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
    │   ├── tabs/                # UploadTab, CleanTab, AnalyzeTab, ModelTab, SuggestTab, ExportTab
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

## API endpoints

```
POST /api/upload
GET  /api/data/preview
POST /api/data/assign_header
GET  /api/data/missing_summary
GET  /api/data/duplicates_summary
POST /api/data/change_type
POST /api/data/rename_column
POST /api/data/clean              fill_missing, drop_rows_with_missing, remove_duplicates,
                                  drop_columns, one_hot_encode, label_encode
GET  /api/data/profile_report
GET  /api/data/correlation_matrix
GET  /api/data/export
POST /api/model/linear_regression  fit_intercept, test_size
POST /api/model/decision_tree      max_depth, min_samples_split, criterion, test_size
POST /api/model/knn                n_neighbors, weights, metric, test_size
POST /api/suggest                  target_column, task_type → Gemini structured JSON
GET  /api/health
```

Each model endpoint returns `metrics`, optional `viz_base64` (2D plot when ≤ 2 features), and `tree_base64` (Decision Tree only).

---

## Notes

- Text columns (object dtype) must be encoded in the **Clean tab** before using as model features.
- Profile report generation can take 30–60 s on large datasets.
- One dataset in memory at a time. Uploading a new file replaces the current one.
- Correlation matrix and profile report reset automatically when the dataset is cleaned or replaced.
