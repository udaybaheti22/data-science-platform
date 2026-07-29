import os
import io
import base64
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CORS — origins read exclusively from the environment variable
# ---------------------------------------------------------------------------

_origins_env = os.getenv("CORS_ORIGINS", "")
cors_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

# If CORS_ORIGINS is set to "*" or left empty, allow all origins
allow_all_origins = not cors_origins or cors_origins == ["*"]

app = FastAPI(title="ML Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_origins,
    allow_credentials=False if allow_all_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Reports directory
# ---------------------------------------------------------------------------

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

data_store: dict = {
    "original_df": None,   # pd.DataFrame | None — immutable, set on upload
    "main_df":     None,   # pd.DataFrame | None — working copy
    "logs":        [],     # list[dict] — action log entries
}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_action(action: str, details: dict | None = None) -> None:
    entry = {"action": action, "details": details or {}}
    data_store["logs"].append(entry)
    logger.info("Action: %s | Details: %s", action, details)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CleanOperation(BaseModel):
    type: str
    columns: list[str] | None = None
    method: str | None = None


class CleanRequest(BaseModel):
    operations: list[CleanOperation]


class ChangeTypeRequest(BaseModel):
    column_name: str
    new_type: str


class RenameColumnRequest(BaseModel):
    old_column_name: str
    new_column_name: str


class LinearRegressionRequest(BaseModel):
    target_column: str
    feature_columns: list[str]
    test_size: float = 0.2
    random_state: int = 42
    fit_intercept: bool = True


class DecisionTreeRequest(BaseModel):
    target_column: str
    feature_columns: list[str]
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int | None = None
    min_samples_split: int = 2
    criterion: str = "gini"


class KNNRequest(BaseModel):
    target_column: str
    feature_columns: list[str]
    test_size: float = 0.2
    random_state: int = 42
    n_neighbors: int = 5
    weights: str = "uniform"
    metric: str = "euclidean"


class AISuggestRequest(BaseModel):
    target_column: str
    task_type: str  # "regression" or "classification"


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate CSV extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()

    # Validate non-empty
    if not contents.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file contains no data rows.")

    data_store["original_df"] = df.copy()
    data_store["main_df"] = df.copy()
    data_store["logs"] = []

    log_action("upload", {"filename": file.filename, "rows": len(df), "columns": len(df.columns)})

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
    }


# ---------------------------------------------------------------------------
# Placeholder routes — full implementations added in subsequent tasks
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Preview — GET /api/data/preview
# ---------------------------------------------------------------------------

@app.get("/api/data/preview")
def get_preview():
    original_df: pd.DataFrame | None = data_store["original_df"]
    if original_df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    total_rows = len(original_df)
    total_columns = len(original_df.columns)
    preview_rows = min(50, total_rows)

    preview_df = original_df.head(preview_rows)
    # Replace NaN values with the string "NaN"
    data = preview_df.where(preview_df.notna(), other="NaN").to_dict(orient="records")

    return {
        "data": data,
        "columns": list(original_df.columns),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "preview_rows": preview_rows,
    }


# ---------------------------------------------------------------------------
# Assign Header — POST /api/data/assign_header
# ---------------------------------------------------------------------------

@app.post("/api/data/assign_header")
def assign_header():
    if data_store["original_df"] is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    main_df: pd.DataFrame = data_store["main_df"]

    if len(main_df) < 2:
        raise HTTPException(status_code=400, detail="Not enough rows to assign header.")

    # Promote the first row to column names
    main_df.columns = [str(v) for v in main_df.iloc[0]]
    # Drop the first row and reset index
    main_df = main_df.iloc[1:].reset_index(drop=True)

    data_store["main_df"] = main_df

    new_columns = list(main_df.columns)
    log_action("assign_header", {"columns": new_columns})

    return {
        "columns": new_columns,
        "rows": len(main_df),
    }


# ---------------------------------------------------------------------------
# POST /api/data/change_type
# ---------------------------------------------------------------------------

VALID_TYPES = {"int64", "float64", "object", "bool", "datetime64[ns]"}


@app.post("/api/data/change_type")
def change_type(request: ChangeTypeRequest):
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    column_name = request.column_name
    new_type = request.new_type

    if column_name not in main_df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found.")

    if new_type not in VALID_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid type '{new_type}'. Valid types: {sorted(VALID_TYPES)}",
        )

    try:
        converted = main_df[column_name].astype(new_type)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot convert column '{column_name}' to {new_type}: {str(e)}",
        )

    data_store["main_df"][column_name] = converted

    log_action("change_type", {"column": column_name, "new_type": new_type})

    updated_df = data_store["main_df"]
    return {
        "message": f"Column '{column_name}' converted to {new_type}.",
        "rows": int(updated_df.shape[0]),
        "columns": int(updated_df.shape[1]),
        "column_list": list(updated_df.columns),
    }


# ---------------------------------------------------------------------------
# POST /api/data/rename_column
# ---------------------------------------------------------------------------

@app.post("/api/data/rename_column")
def rename_column(request: RenameColumnRequest):
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    old_name = request.old_column_name
    new_name = request.new_column_name

    if old_name not in main_df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{old_name}' not found.")

    updated_df = main_df.rename(columns={old_name: new_name})
    data_store["main_df"] = updated_df

    log_action("rename_column", {"old_name": old_name, "new_name": new_name})

    return {
        "message": f"Column '{old_name}' renamed to '{new_name}'.",
        "rows": int(updated_df.shape[0]),
        "columns": int(updated_df.shape[1]),
        "column_list": list(updated_df.columns),
    }

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

@app.get("/api/data/missing_summary")
def get_missing_summary():
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    return [
        {"column_name": col, "missing_count": int(main_df[col].isnull().sum())}
        for col in main_df.columns
    ]


@app.get("/api/data/duplicates_summary")
def get_duplicates_summary():
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    return {"duplicate_count": int(main_df.duplicated().sum())}


# ---------------------------------------------------------------------------
# POST /api/data/clean
# ---------------------------------------------------------------------------

@app.post("/api/data/clean")
def clean_data(request: CleanRequest):
    if data_store["main_df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    df = data_store["main_df"].copy()
    applied = []

    for op in request.operations:
        op_type = op.type

        if op_type == "fill_missing":
            if not op.columns:
                raise HTTPException(status_code=400, detail="fill_missing requires 'columns'.")
            if not op.method:
                raise HTTPException(status_code=400, detail="fill_missing requires 'method'.")
            for col in op.columns:
                if col not in df.columns:
                    raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
            for col in op.columns:
                if op.method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif op.method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif op.method == "mode":
                    mode_vals = df[col].mode()
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals[0])
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown fill method '{op.method}'. Use mean, median, or mode.",
                    )
            applied.append({"type": op_type, "columns": op.columns, "method": op.method})

        elif op_type == "drop_rows_with_missing":
            subset = op.columns if op.columns else None
            df = df.dropna(subset=subset)
            applied.append({"type": op_type, "columns": op.columns})

        elif op_type == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df)
            applied.append({"type": op_type, "rows_removed": removed})

        elif op_type == "drop_columns":
            if not op.columns:
                raise HTTPException(status_code=400, detail="drop_columns requires 'columns'.")
            for col in op.columns:
                if col not in df.columns:
                    raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
            df = df.drop(columns=op.columns)
            applied.append({"type": op_type, "columns": op.columns})

        elif op_type == "one_hot_encode":
            if not op.columns:
                raise HTTPException(status_code=400, detail="one_hot_encode requires 'columns'.")
            for col in op.columns:
                if col not in df.columns:
                    raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
            df = pd.get_dummies(df, columns=op.columns)
            applied.append({"type": op_type, "columns": op.columns})

        elif op_type == "label_encode":
            if not op.columns:
                raise HTTPException(status_code=400, detail="label_encode requires 'columns'.")
            for col in op.columns:
                if col not in df.columns:
                    raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")
            for col in op.columns:
                df[col] = df[col].astype("category").cat.codes
            applied.append({"type": op_type, "columns": op.columns})

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation type '{op_type}'.")

    data_store["main_df"] = df
    log_action("clean", {"operations": applied})

    return {
        "message": "Operations applied successfully.",
        "rows": len(df),
        "columns": len(df.columns),
        "column_list": df.columns.tolist(),
    }


# ---------------------------------------------------------------------------
# GET /api/data/profile_report
# ---------------------------------------------------------------------------

@app.get("/api/data/profile_report")
def get_profile_report():
    if data_store["main_df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            data_store["main_df"],
            title="Dataset Profile",
            explorative=False,   # explorative=True triggers Numba which requires NumPy ≤2.3
            progress_bar=False,
        )
        filename = f"profile_{uuid.uuid4().hex[:8]}.html"
        profile.to_file(REPORTS_DIR / filename)
        return {"report_url": f"/api/reports/{filename}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating profile report: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /api/reports/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/reports/{filename}")
def serve_report(filename: str):
    file_path = REPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(str(file_path), media_type="text/html")


# ---------------------------------------------------------------------------
# ML Helper Functions
# ---------------------------------------------------------------------------

def detect_task_type(y: pd.Series) -> str:
    """Return 'regression' for float64 columns, 'classification' otherwise."""
    if y.dtype == np.float64:
        return "regression"
    return "classification"


def compute_regression_metrics(y_test, y_pred) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compute_classification_metrics(y_test, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def shared_train_guards(
    target_column: str,
    feature_columns: list[str],
    test_size: float,
    random_state: int,
):
    """
    Shared validation and data preparation for all model training endpoints.
    Returns (X_train, X_test, y_train, y_test, dropped_rows) on success.
    Raises HTTPException on any guard failure.
    """
    from sklearn.model_selection import train_test_split

    df = data_store["main_df"]
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    if test_size < 0.1:
        raise HTTPException(status_code=400, detail="test_size must be at least 0.1.")

    if target_column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Target column '{target_column}' not found.")

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise HTTPException(
            status_code=404,
            detail=f"Feature column(s) not found: {missing_features}",
        )

    # Validate all columns are numeric
    all_cols = feature_columns + [target_column]
    non_numeric = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise HTTPException(
            status_code=400,
            detail=f"Non-numeric columns cannot be used as features: {non_numeric}",
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Drop rows where features have NaN
    before = len(X)
    mask = X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    dropped_rows = before - len(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, dropped_rows


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------

def generate_correlation_matrix(df: pd.DataFrame) -> str:
    """
    Compute the Pearson correlation matrix for all numeric columns,
    render it as a matplotlib heatmap with annotated values,
    and return a base64-encoded PNG string.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty or numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for a correlation matrix.")

    corr = numeric_df.corr()
    n = len(corr.columns)
    fig_size = max(6, n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_title("Pearson Correlation Matrix", fontsize=12, pad=10)
    fig.tight_layout()
    return _fig_to_base64(fig)


@app.get("/api/data/correlation_matrix")
def get_correlation_matrix():
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        img_base64 = generate_correlation_matrix(main_df)
        return {"img_base64": img_base64}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating correlation matrix: {str(e)}")


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Save a matplotlib figure to a base64-encoded PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_viz(model, X_test, y_test, y_pred, feature_columns: list[str], model_name: str) -> str | None:
    """
    Generate a 2D visualization PNG (base64) for <= 2 feature columns.
    Returns None when more than 2 features are present.
    """
    n_features = len(feature_columns)
    if n_features > 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    if model_name == "linear_regression":
        if n_features == 1:
            x_vals = X_test.iloc[:, 0].values
            sort_idx = np.argsort(x_vals)
            ax.scatter(x_vals, y_test, color="steelblue", alpha=0.7, label="Actual")
            ax.plot(x_vals[sort_idx], y_pred[sort_idx], color="red", linewidth=2, label="Regression line")
            ax.set_xlabel(feature_columns[0])
            ax.set_ylabel("Target")
            ax.legend()
            ax.set_title(f"Linear Regression: {feature_columns[0]} vs Target")
        else:  # n_features == 2
            sc = ax.scatter(
                X_test.iloc[:, 0], X_test.iloc[:, 1],
                c=y_test, cmap="viridis", alpha=0.8
            )
            fig.colorbar(sc, ax=ax, label="Target value")
            ax.set_xlabel(feature_columns[0])
            ax.set_ylabel(feature_columns[1])
            ax.set_title("Feature Scatter (colored by target)")

    elif model_name in ("decision_tree", "knn"):
        try:
            x0 = X_test.iloc[:, 0].values
            x0_min, x0_max = x0.min() - 1, x0.max() + 1

            if n_features == 2:
                x1 = X_test.iloc[:, 1].values
                x1_min, x1_max = x1.min() - 1, x1.max() + 1
            else:
                x1_min, x1_max = -1.0, 1.0

            xx, yy = np.meshgrid(
                np.linspace(x0_min, x0_max, 150),
                np.linspace(x1_min, x1_max, 150),
            )

            if n_features == 1:
                grid_input = np.c_[xx.ravel(), np.zeros_like(xx.ravel())]
            else:
                grid_input = np.c_[xx.ravel(), yy.ravel()]

            # Use only the feature columns used during training
            import pandas as _pd
            grid_df = _pd.DataFrame(grid_input, columns=feature_columns if n_features == 2 else [feature_columns[0], "__dummy__"])
            grid_df = grid_df[feature_columns]

            Z = model.predict(grid_df)
            # Convert to numeric for contour plotting
            if Z.dtype == object or str(Z.dtype) == "category":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                Z = le.fit_transform(Z).astype(float)
            else:
                Z = Z.astype(float)

            ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.35, cmap="RdYlBu")

            scatter_x = X_test.iloc[:, 0].values
            scatter_y = X_test.iloc[:, 1].values if n_features == 2 else np.zeros_like(scatter_x)
            y_test_num = y_test.values.astype(float) if hasattr(y_test, "values") else np.array(y_test, dtype=float)
            ax.scatter(scatter_x, scatter_y, c=y_test_num, cmap="RdYlBu", edgecolors="k", s=40)
            ax.set_xlabel(feature_columns[0])
            if n_features == 2:
                ax.set_ylabel(feature_columns[1])
            ax.set_title(f"{model_name.replace('_', ' ').title()} Decision Boundary")
        except Exception:
            plt.close(fig)
            return None

    return _fig_to_base64(fig)


def generate_tree_diagram(model, feature_names: list[str]) -> str:
    """
    Generate a full sklearn decision tree diagram (plot_tree) as a base64 PNG.
    Always generated for Decision Tree models regardless of feature count.
    """
    import sklearn.tree as _sk_tree
    n_features = len(feature_names)
    # Scale figure size to depth/features so it remains readable
    fig_width = max(20, n_features * 5)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    _sk_tree.plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure", fontsize=14)
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# POST /api/model/linear_regression
# ---------------------------------------------------------------------------

@app.post("/api/model/linear_regression")
def train_linear_regression(request: LinearRegressionRequest):
    from sklearn.linear_model import LinearRegression as _LR

    X_train, X_test, y_train, y_test, dropped_rows = shared_train_guards(
        request.target_column,
        request.feature_columns,
        request.test_size,
        request.random_state,
    )

    model = _LR(fit_intercept=request.fit_intercept)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_regression_metrics(y_test, y_pred)
    viz = generate_viz(model, X_test, y_test, y_pred, request.feature_columns, "linear_regression")

    log_action("train_linear_regression", {"features": request.feature_columns, "target": request.target_column})

    return {
        "model_name": "linear_regression",
        "task_type": "regression",
        "metrics": metrics,
        "dropped_rows": dropped_rows,
        "viz_base64": viz,
        "tree_base64": None,
    }


# ---------------------------------------------------------------------------
# POST /api/model/decision_tree
# ---------------------------------------------------------------------------

@app.post("/api/model/decision_tree")
def train_decision_tree(request: DecisionTreeRequest):
    from sklearn.tree import DecisionTreeClassifier as _DTC, DecisionTreeRegressor as _DTR

    X_train, X_test, y_train, y_test, dropped_rows = shared_train_guards(
        request.target_column,
        request.feature_columns,
        request.test_size,
        request.random_state,
    )

    task_type = detect_task_type(y_train)

    if task_type == "classification":
        criterion = request.criterion if request.criterion in ("gini", "entropy") else "gini"
        model = _DTC(
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            criterion=criterion,
            random_state=request.random_state,
        )
    else:
        criterion = request.criterion if request.criterion in ("squared_error", "absolute_error") else "squared_error"
        model = _DTR(
            max_depth=request.max_depth,
            min_samples_split=request.min_samples_split,
            criterion=criterion,
            random_state=request.random_state,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metrics = compute_classification_metrics(y_test, y_pred)
    else:
        metrics = compute_regression_metrics(y_test, y_pred)

    viz = generate_viz(model, X_test, y_test, y_pred, request.feature_columns, "decision_tree")
    tree_diagram = generate_tree_diagram(model, request.feature_columns)

    log_action("train_decision_tree", {"features": request.feature_columns, "target": request.target_column})

    return {
        "model_name": "decision_tree",
        "task_type": task_type,
        "metrics": metrics,
        "dropped_rows": dropped_rows,
        "viz_base64": viz,
        "tree_base64": tree_diagram,
    }


# ---------------------------------------------------------------------------
# POST /api/model/knn
# ---------------------------------------------------------------------------

@app.post("/api/model/knn")
def train_knn(request: KNNRequest):
    from sklearn.neighbors import KNeighborsClassifier as _KNNC, KNeighborsRegressor as _KNNR

    X_train, X_test, y_train, y_test, dropped_rows = shared_train_guards(
        request.target_column,
        request.feature_columns,
        request.test_size,
        request.random_state,
    )

    task_type = detect_task_type(y_train)

    if task_type == "classification":
        model = _KNNC(
            n_neighbors=request.n_neighbors,
            weights=request.weights,
            metric=request.metric,
        )
    else:
        model = _KNNR(
            n_neighbors=request.n_neighbors,
            weights=request.weights,
            metric=request.metric,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metrics = compute_classification_metrics(y_test, y_pred)
    else:
        metrics = compute_regression_metrics(y_test, y_pred)

    viz = generate_viz(model, X_test, y_test, y_pred, request.feature_columns, "knn")

    log_action("train_knn", {"features": request.feature_columns, "target": request.target_column})

    return {
        "model_name": "knn",
        "task_type": task_type,
        "metrics": metrics,
        "dropped_rows": dropped_rows,
        "viz_base64": viz,
        "tree_base64": None,
    }


# ---------------------------------------------------------------------------
# POST /api/suggest — AI-powered suggestions using Gemini
# ---------------------------------------------------------------------------

@app.post("/api/suggest")
def get_ai_suggestions(request: AISuggestRequest):
    """
    Generate AI suggestions for data cleaning and model selection using Gemini.
    Returns structured JSON with column_suggestions and model_suggestions.
    """
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    target_column = request.target_column
    task_type = request.task_type.lower()

    if target_column not in main_df.columns:
        raise HTTPException(status_code=404, detail=f"Target column '{target_column}' not found.")

    if task_type not in ("regression", "classification"):
        raise HTTPException(status_code=400, detail="task_type must be 'regression' or 'classification'.")

    # Build compact dataset profile
    profile_lines = []
    profile_lines.append(f"Dataset: {len(main_df)} rows × {len(main_df.columns)} columns")
    profile_lines.append(f"Target column: {target_column}")
    profile_lines.append(f"Task type: {task_type}")
    profile_lines.append("")
    profile_lines.append("Column profiles:")

    for col in main_df.columns:
        dtype = str(main_df[col].dtype)
        null_pct = (main_df[col].isnull().sum() / len(main_df)) * 100
        unique_count = main_df[col].nunique()

        line = f"- {col}: dtype={dtype}, null={null_pct:.1f}%, unique={unique_count}"

        if pd.api.types.is_numeric_dtype(main_df[col]):
            non_null = main_df[col].dropna()
            if len(non_null) > 0:
                line += f", min={non_null.min():.2f}, max={non_null.max():.2f}"
        else:
            top_vals = main_df[col].value_counts().head(5).to_dict()
            line += f", top_5={list(top_vals.keys())}"

        profile_lines.append(line)

    # Compute correlation with target (encode target if classification)
    profile_lines.append("")
    profile_lines.append(f"Pearson correlation with target '{target_column}':")

    try:
        if task_type == "classification":
            # Label encode target temporarily for correlation
            target_encoded = main_df[target_column].astype("category").cat.codes
        else:
            target_encoded = main_df[target_column]

        for col in main_df.columns:
            if col == target_column:
                continue
            if pd.api.types.is_numeric_dtype(main_df[col]):
                non_null_mask = main_df[col].notna() & target_encoded.notna()
                if non_null_mask.sum() > 1:
                    corr = main_df.loc[non_null_mask, col].corr(target_encoded[non_null_mask])
                    profile_lines.append(f"- {col}: {corr:.3f}")
    except Exception:
        profile_lines.append("(correlation computation failed)")

    dataset_profile = "\n".join(profile_lines)

    # Call Gemini API with structured output
    try:
        from google import genai
        from google.genai import types
        from pydantic import BaseModel as PydanticBaseModel
        from typing import List

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY environment variable not set.",
            )

        from typing import Optional as Opt

        # Pydantic models for structured output (google-genai SDK)
        # NOTE: No dict/additionalProperties — Developer API doesn't support it.
        # All hyperparameter fields are optional so they work across all 3 models.
        class ColumnSuggestion(PydanticBaseModel):
            column: str
            action: str
            reason: str

        class SuggestedHyperparameters(PydanticBaseModel):
            # linear_regression
            fit_intercept: Opt[bool] = None
            # decision_tree
            max_depth: Opt[int] = None
            min_samples_split: Opt[int] = None
            criterion: Opt[str] = None
            # knn
            n_neighbors: Opt[int] = None
            weights: Opt[str] = None
            metric: Opt[str] = None
            # shared
            test_size: Opt[float] = None

        class ModelSuggestions(PydanticBaseModel):
            recommended_model: str
            hyperparameters: SuggestedHyperparameters

        class AISuggestionResult(PydanticBaseModel):
            summary_text: str
            target_column: str
            task_type: str
            column_suggestions: List[ColumnSuggestion]
            model_suggestions: ModelSuggestions

        client = genai.Client(api_key=api_key)

        prompt = f"""You are an expert data scientist analyzing a dataset for machine learning.

Dataset Profile:
{dataset_profile}

Task: Provide actionable suggestions for data cleaning and model selection.

CRITICAL CONSTRAINTS:
1. For column_suggestions, ONLY use these actions: fill_missing, drop_rows_with_missing, drop_columns, one_hot_encode, label_encode, change_type, rename_column, remove_duplicates
2. For model_suggestions.recommended_model, ONLY use: linear_regression, decision_tree, knn
3. For hyperparameters, ONLY include parameters that exist for that model:
   - linear_regression: fit_intercept (bool), test_size (0.1-0.5)
   - decision_tree: max_depth (int or null), min_samples_split (int >= 2), criterion (gini/entropy for classification OR squared_error/absolute_error for regression), test_size (0.1-0.5)
   - knn: n_neighbors (int >= 1), weights (uniform or distance), metric (euclidean, manhattan, or minkowski), test_size (0.1-0.5)

Focus on:
- Columns with high null % that need filling or dropping
- Text columns (dtype=object) that need encoding before ML
- Highly correlated features for the selected model
- Reasonable hyperparameter defaults (e.g., test_size around 0.2, n_neighbors between 3-7, max_depth between 3-10 or null)

Return structured JSON matching the schema exactly."""

        import json
        import time

        # Model fallback list — try primary first, fall back on 503/429
        model_candidates = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.0-flash-lite"]
        response = None
        last_error = None

        for model_name in model_candidates:
            for attempt in range(2):  # 2 attempts per model before moving to next
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=AISuggestionResult,
                        ),
                    )
                    logger.info(f"Gemini responded OK with model: {model_name}")
                    break  # success
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    is_transient = any(code in err_str for code in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])
                    if is_transient and attempt == 0:
                        logger.warning(f"Gemini {model_name} attempt 1 failed ({err_str[:60]}), retrying in 2s...")
                        time.sleep(2)
                    else:
                        logger.warning(f"Gemini {model_name} failed, trying next model...")
                        break  # move to next model
            if response is not None:
                break  # got a response, stop trying models

        if response is None:
            raise last_error
        parsed = json.loads(response.text)

        # Strip None hyperparameter fields so the frontend only sees relevant ones
        if "model_suggestions" in parsed and "hyperparameters" in parsed["model_suggestions"]:
            parsed["model_suggestions"]["hyperparameters"] = {
                k: v for k, v in parsed["model_suggestions"]["hyperparameters"].items()
                if v is not None
            }

        log_action("ai_suggest", {"target": target_column, "task_type": task_type})

        return parsed

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-genai package not installed. Run: pip install google-genai",
        )
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"AI suggestion generation failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /api/data/export
# ---------------------------------------------------------------------------

@app.get("/api/data/export")
def export_dataset():
    main_df = data_store["main_df"]
    if main_df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    def csv_streamer():
        yield main_df.to_csv(index=False)

    return StreamingResponse(
        csv_streamer(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"},
    )
