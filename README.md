# Data Science / EDA / ML Workbench

A lightweight workbench for uploading, cleaning, profiling, and exporting datasets with a simple browser UI and a FastAPI backend.

## Refactor Notice

This repository has been structurally refactored. Functionality and logic remain the same. The undo feature is temporarily disabled during this refactor phase.

## Current Frontend (Vanilla HTML / CSS / JS)

The frontend is now a **single-page dashboard** built with plain HTML, CSS, and JavaScript (no React / Tailwind / Bootstrap). It uses a fixed left sidebar and card-based content area.

### Sidebar Sections

The sidebar contains the following sections. Only features backed by existing backend APIs are wired up; others are clearly marked as placeholders.

- **Preview** (functional)
  - Upload CSV/XLS(X) using the **Upload Dataset** card
  - Calls `POST /api/upload`, then `GET /api/data/preview?limit=50`
  - Renders a table preview and basic metadata (rows / columns)

- **Data** (functional)
  - **Column Types Management** table with per-row data type conversion
  - Allows changing column data types with inline error handling
  - Each row has a dropdown for target type and Save button

- **Clean** (partially functional)
  - **Missing Values** card:
    - On navigation to **Clean**, calls `GET /api/data/missing_summary`
    - Renders per-column missing counts/ratios in a list
  - All other clean operations (duplicates, drop columns, encoding, scaling) are **UI-only placeholders** until their APIs are implemented

- **Analyze** (functional)
  - **Generate Profile Report** button calls `GET /api/data/profile_report`
  - Opens comprehensive HTML profiling report in new browser tab (Jupyter-like behavior)

- **Build Model** (placeholder)
  - Describes a future modeling workspace
  - No backend calls

- **Logs** (placeholder)
  - Static descriptive text; no log API is called yet

- **Export** (functional)
  - **Export Dataset** button uses `GET /api/data/export?format=csv|parquet`
  - Triggers a file download (`dataset.csv` or `dataset.parquet`)

Error handling remains **explicit**: backend errors are surfaced in a top-of-page alert and logged to the browser console.

## Tech Stack

- **Backend**: FastAPI, Python, Pandas, NumPy, scikit-learn, Matplotlib
- **Frontend**: HTML, CSS, Vanilla JavaScript

## Project Structure

```text
frontend/
├── index.html
└── static/
    ├── css/
    │   ├── main.css       # Layout, typography, buttons, tables, section layout
    │   ├── sidebar.css    # Fixed sidebar look & feel
    │   └── cards.css      # Card shadow, radius, border, spacing
    ├── js/
    │   ├── api.js         # Thin wrappers around existing FastAPI endpoints
    │   ├── ui.js          # Section switching, sidebar active state, alerts
    │   ├── render.js      # DOM rendering helpers (tables, lists, logs)
    │   └── events.js      # Wiring of buttons, sidebar, and API calls
    └── icons/             # Small SVG icons for sidebar items

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

```bash
cd backend
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

## How to Run the Frontend

- Open `frontend/index.html` directly in your browser, or
- Serve `frontend/` via a static server (e.g. VS Code Live Server or `python -m http.server`)

Because the frontend is pure HTML/JS, there is **no build step**.

## Effective Single-Page Workflow

- **1. Upload**: Go to **Preview**, upload a dataset, and confirm that the preview table and metadata load.
- **2. Profile**: Use **Analyze** to generate an HTML profile report that opens in a new browser tab.
- **3. Data Types**: Navigate to **Data** to see column types and change them using the per-row dropdowns and Save buttons.
- **4. Inspect Missingness**: Navigate to **Clean** to see the Missing Values summary (backed by `/api/data/missing_summary`).
- **5. Export**: Use **Export** to download the current dataset in CSV or Parquet format.

All other UI elements are designed and styled but intentionally **do not** call the backend until the corresponding APIs are implemented.

## Current Limitations

- In-memory storage only; no database
- Many cleaning and modeling controls are visual placeholders pending backend support
