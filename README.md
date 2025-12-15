# Data Science / EDA / ML Workbench

A lightweight workbench for uploading, cleaning, profiling, and exporting datasets with a simple browser UI and a FastAPI backend.

## Refactor Notice

This repository has been structurally refactored. Functionality and logic remain the same. The undo feature is temporarily disabled during this refactor phase.

## Iteration 2: Single-Page Dashboard

This iteration introduces a single-page dashboard focused on visibility and debuggability. Sections:

- Upload (functional): select file and upload, view preview and metadata
- Clean (placeholder): visible shell, features enabled progressively
- Analyze (functional): generate data profile and download HTML
- Model (placeholder): to be enabled later
- Export (functional): download dataset as CSV

Error handling is loud by design: backend errors are surfaced directly in the UI and logged to the console.

## Iteration 3: UI/UX + Cleaning + Checkpointing

Enhancements:

- Sidebar-based navigation on a single page; jumps to sections and highlights the active item
- Clean section provides stacked, scroll-through operations:
  - Missing values: drop rows/columns; fill by mean/median/mode/drop/constant
  - Duplicates: remove duplicate rows
  - Column operations: drop columns; rename column
  - Data types: convert column dtype (`int64`, `float64`, `object`, `datetime64[ns]`)
  - Encoding: label and one-hot encoding
  - Scaling: StandardScaler and MinMaxScaler
  - Outliers: UI only (not implemented)
- Dataset checkpointing:
  - Explicit Save Dataset button
  - Max 5 checkpoints with FIFO eviction
  - Confirmation prompt when limit reached
  - Export allows choosing final dataset or a checkpoint

Strict error-first behavior preserved.

## Tech Stack

- FastAPI, Python, Pandas, NumPy, scikit-learn, Matplotlib
- HTML, CSS, Vanilla JavaScript

## Project Structure

```
frontend/
├── index.html
└── static/
    ├── css/
    │   └── main.css
    └── js/
        ├── api.js
        ├── events.js
        ├── render.js
        └── main.js

backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── datastore.py
│   │   └── logger.py
│   └── models/
│       └── schemas.py
└── requirements.txt
```

## How to Run the Backend

```
cd backend
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

## How to Run the Frontend

- Open `frontend/index.html` directly in your browser, or
- Use a static server (e.g., VS Code Live Server or `python -m http.server`)

## Single-Page Workflow

- Use the top buttons to switch sections; all content lives on one page.
- Upload shows a table preview and dataset metadata.
- Analyze generates an HTML report via the backend and triggers a download.
- Export downloads the current dataset as CSV.
 - Clean provides stacked blocks to apply operations; updates preview on success.
 - Save Dataset stores the current DataFrame as a checkpoint (max 5).

## Functional vs Placeholder

- Functional: Upload, Clean operations, Analyze (profile download), Export (CSV and checkpoints)
- Placeholder: Model

## Current Limitations

- No database; in-memory storage only
- Undo disabled during refactor
