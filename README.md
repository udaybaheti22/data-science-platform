# Data Science / EDA / ML Workbench

A lightweight workbench for uploading, cleaning, profiling, and exporting datasets with a simple browser UI and a FastAPI backend.

## Refactor Notice

This repository has been structurally refactored. Functionality and logic remain the same. The undo feature is temporarily disabled during this refactor phase.

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

## Current Limitations

- No database; in-memory storage only
- Undo disabled during refactor
