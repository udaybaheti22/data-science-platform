import io
import json
import base64
from io import BytesIO
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .core.datastore import data_store
from .core.logger import log_action
from .models.schemas import TrainRequest


app = FastAPI(
    title="Exploratory Data Analysis API",
    description="An API for performing EDA on uploaded datasets."
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:8000",
    "*",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory at /app
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename or ""
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or Excel.")
        data_store["main_df"] = df
        log_action(f"Uploaded dataset: {filename} ({len(df)} rows, {len(df.columns)} columns)")
        return {"filename": filename, "rows": len(df), "columns": len(df.columns)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


@app.get("/api/data/preview")
async def get_data_preview(limit: int = 50):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    preview_df = df.head(limit)
    preview_data = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            row_dict[col] = "NaN" if pd.isna(value) else value
        preview_data.append(row_dict)
    return {
        "data": preview_data,
        "columns": df.columns.tolist(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "preview_rows": len(preview_df),
    }


@app.get("/api/data/duplicates_summary")
async def get_duplicates_summary():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        duplicate_count = df.duplicated().sum()
        return {"duplicate_count": int(duplicate_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating duplicates: {e}")


@app.post("/api/data/clean")
async def clean_dataset(cleaning_operations: Dict[str, Any]):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        # data_store["history"].append(df.copy())
        modified_df = df.copy()
        operations = cleaning_operations.get("operations", [])
        for operation in operations:
            op_type = operation.get("type")
            if op_type == "drop_columns":
                columns_to_drop = operation.get("columns", [])
                modified_df = modified_df.drop(columns=columns_to_drop, errors="ignore")
                log_action(f"Dropped columns: {', '.join(columns_to_drop)}")
            elif op_type == "drop_rows_with_missing":
                threshold = operation.get("threshold", 0.5)
                modified_df = modified_df.dropna(thresh=len(modified_df.columns) * threshold)
                log_action(f"Dropped rows with missing values (threshold: {threshold})")
            elif op_type == "fill_missing":
                method = operation.get("method", "mean")
                columns = operation.get("columns", [])
                constant_value = operation.get("value")
                for col in columns:
                    if col in modified_df.columns:
                        if method == "mean" and pd.api.types.is_numeric_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].fillna(modified_df[col].mean())
                        elif method == "median" and pd.api.types.is_numeric_dtype(modified_df[col]):
                            modified_df[col] = modified_df[col].fillna(modified_df[col].median())
                        elif method == "mode":
                            modified_df[col] = modified_df[col].fillna(modified_df[col].mode()[0] if not modified_df[col].mode().empty else "Unknown")
                        elif method == "drop":
                            modified_df = modified_df.dropna(subset=[col])
                        elif method == "constant":
                            if constant_value is None:
                                raise HTTPException(status_code=422, detail="Constant value is required for fill_missing with method 'constant'")
                            modified_df[col] = modified_df[col].fillna(constant_value)
                log_action(f"Filled missing values in columns: {', '.join(columns)} using {method} method")
            elif op_type == "remove_duplicates":
                limit = operation.get("limit", None)
                if limit is None:
                    original_count = len(modified_df)
                    modified_df = modified_df.drop_duplicates()
                    removed_count = original_count - len(modified_df)
                    log_action(f"Removed {removed_count} duplicate rows.")
                else:
                    duplicate_indices = modified_df[modified_df.duplicated()].index
                    if len(duplicate_indices) > 0:
                        indices_to_drop = duplicate_indices[:limit]
                        modified_df = modified_df.drop(indices_to_drop)
                        log_action(f"Removed {len(indices_to_drop)} duplicate rows (limited).")
                    else:
                        log_action("No duplicate rows found to remove.")
            elif op_type == "one_hot_encode":
                columns_to_encode = operation.get("columns", [])
                if not columns_to_encode:
                    raise HTTPException(status_code=400, detail="No columns specified for one-hot encoding")
                missing_columns = [col for col in columns_to_encode if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                encoded_df = pd.get_dummies(modified_df, columns=columns_to_encode)
                modified_df = encoded_df
                log_action(f"One-Hot Encoded columns: {', '.join(columns_to_encode)}")
            elif op_type == "label_encode":
                columns_to_encode = operation.get("columns", [])
                if not columns_to_encode:
                    raise HTTPException(status_code=400, detail="No columns specified for label encoding")
                missing_columns = [col for col in columns_to_encode if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                for column_name in columns_to_encode:
                    if column_name in modified_df.columns:
                        encoder = LabelEncoder()
                        modified_df[column_name] = encoder.fit_transform(modified_df[column_name])
                log_action(f"Label Encoded columns: {', '.join(columns_to_encode)}")
            elif op_type == "scale":
                columns_to_scale = operation.get("columns", [])
                method = operation.get("method", "standard")
                if not columns_to_scale:
                    raise HTTPException(status_code=400, detail="No columns specified for scaling")
                missing_columns = [col for col in columns_to_scale if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                non_numerical_columns = [col for col in columns_to_scale if not pd.api.types.is_numeric_dtype(modified_df[col])]
                if non_numerical_columns:
                    raise HTTPException(status_code=400, detail=f"Non-numerical columns cannot be scaled: {', '.join(non_numerical_columns)}")
                if method == "standard":
                    scaler = StandardScaler()
                    modified_df[columns_to_scale] = scaler.fit_transform(modified_df[columns_to_scale])
                    log_action(f"Applied standard scaling to columns: {', '.join(columns_to_scale)}")
                elif method == "minmax":
                    scaler = MinMaxScaler()
                    modified_df[columns_to_scale] = scaler.fit_transform(modified_df[columns_to_scale])
                    log_action(f"Applied minmax scaling to columns: {', '.join(columns_to_scale)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid scaling method: {method}. Use 'standard' or 'minmax'")
        data_store["main_df"] = modified_df
        return {"message": "Dataset cleaned successfully", "rows": len(modified_df), "columns": len(modified_df.columns), "history_length": len(data_store["history"]) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {e}")


@app.post("/api/data/change_type")
async def change_column_type(column_data: Dict[str, str]):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        column_name = column_data.get("column_name")
        new_type = column_data.get("new_type")
        if not column_name or not new_type:
            raise HTTPException(status_code=400, detail="Both column_name and new_type are required")
        if column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found")
        # data_store["history"].append(df.copy())
        try:
            df[column_name] = df[column_name].astype(new_type)
            log_action(f"Changed type of '{column_name}' to '{new_type}'.")
            return {"message": f"Successfully changed type of '{column_name}' to '{new_type}'", "rows": len(df), "columns": len(df.columns), "history_length": len(data_store["history"]) }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid conversion: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error converting column type: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error changing column type: {e}")


@app.post("/api/data/rename_column")
async def rename_column(column_data: Dict[str, str]):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        old_column_name = column_data.get("old_column_name")
        new_column_name = column_data.get("new_column_name")
        if not old_column_name or not new_column_name:
            raise HTTPException(status_code=400, detail="Both old_column_name and new_column_name are required")
        if old_column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{old_column_name}' not found")
        if new_column_name in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{new_column_name}' already exists")
        # data_store["history"].append(df.copy())
        df.rename(columns={old_column_name: new_column_name}, inplace=True)
        log_action(f"Renamed column '{old_column_name}' to '{new_column_name}'.")
        return {"message": f"Successfully renamed column '{old_column_name}' to '{new_column_name}'", "rows": len(df), "columns": len(df.columns), "history_length": len(data_store["history"]) }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming column: {e}")


@app.post("/api/project/save")
async def save_project_state(project_data: Dict[str, Any]):
    try:
        project_id = project_data.get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
        data_store["project_states"][project_id] = project_data
        return {"message": "Project state saved successfully", "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving project state: {e}")


@app.get("/api/project/logs")
async def get_project_logs():
    return {"logs": data_store["logs"]}


@app.get("/api/project/{project_id}")
async def get_project_state(project_id: str):
    project_state = data_store["project_states"].get(project_id)
    if not project_state:
        raise HTTPException(status_code=404, detail="Project not found")
    return project_state


@app.get("/api/data/missing_summary")
async def get_missing_summary():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        missing_counts = df.isnull().sum()
        missing_summary = []
        for column_name, missing_count in missing_counts.items():
            missing_summary.append({"column_name": column_name, "missing_count": int(missing_count)})
        return missing_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating missing values: {e}")


@app.get("/api/data/export")
async def export_dataset(name: str = "full"):
    try:
        if name == "train":
            X_train = data_store.get("X_train")
            y_train = data_store.get("y_train")
            if X_train is None or y_train is None:
                raise HTTPException(status_code=404, detail="Training set not found. Please train a model first.")
            train_df = X_train.copy()
            train_df[data_store.get("target_column", "target")] = y_train
            csv_string = train_df.to_csv(index=False)
            filename = "train_dataset.csv"
        elif name == "test":
            X_test = data_store.get("X_test")
            y_test = data_store.get("y_test")
            if X_test is None or y_test is None:
                raise HTTPException(status_code=404, detail="Test set not found. Please train a model first.")
            test_df = X_test.copy()
            test_df[data_store.get("target_column", "target")] = y_test
            csv_string = test_df.to_csv(index=False)
            filename = "test_dataset.csv"
        else:
            df = data_store.get("main_df")
            if df is None:
                raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
            csv_string = df.to_csv(index=False)
            filename = "dataset.csv"
        csv_bytes = csv_string.encode("utf-8")
        csv_io = io.BytesIO(csv_bytes)
        return StreamingResponse(iter([csv_io.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {e}")


@app.get("/api/data/profile")
async def get_data_profile():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    info_df = pd.DataFrame({"Non-Null Count": df.notna().sum(), "Dtype": df.dtypes.astype(str)}).reset_index().rename(columns={"index": "Column"})
    desc_stats = df.describe().round(3).reset_index().rename(columns={"index": "Statistic"})
    value_counts = {}
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        counts = df[col].value_counts().head(5)
        value_counts[col] = counts.to_dict()
    data_head = []
    for _, row in df.head().iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            row_dict[col] = "NaN" if pd.isna(value) else value
        data_head.append(row_dict)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    return {"data_head": data_head, "column_info": info_df.to_dict(orient="records"), "value_counts": value_counts, "numerical_columns": numerical_cols}


@app.get("/api/data/profile_report")
async def get_profile_report():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        log_action("Generated comprehensive data profiling report.")
        import datetime
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #667eea; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
                .metric .value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class=\"header\">
                <h1>📊 Data Profile Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class=\"section\">
                <h2>📈 Dataset Overview</h2>
                <div class=\"metric\"><h3>Total Rows</h3><div class=\"value\">{len(df):,}</div></div>
                <div class=\"metric\"><h3>Total Columns</h3><div class=\"value\">{len(df.columns)}</div></div>
                <div class=\"metric\"><h3>Memory Usage</h3><div class=\"value\">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div></div>
                <div class=\"metric\"><h3>Missing Values</h3><div class=\"value\">{df.isnull().sum().sum():,}</div></div>
            </div>
        """
        html_content += """
            <div class=\"section\">
                <h2>🔍 Column Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Data Type</th>
                            <th>Non-Null Count</th>
                            <th>Missing Values</th>
                            <th>Unique Values</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        for col in df.columns:
            html_content += f"""
                        <tr>
                            <td><strong>{col}</strong></td>
                            <td>{df[col].dtype}</td>
                            <td>{df[col].notna().sum():,}</td>
                            <td>{df[col].isnull().sum():,}</td>
                            <td>{df[col].nunique():,}</td>
                        </tr>
            """
        html_content += """
                    </tbody>
                </table>
            </div>
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            html_content += """
            <div class=\"section\">
                <h2>📊 Numerical Columns Statistics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>50%</th>
                            <th>75%</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for col in numerical_cols:
                stats = df[col].describe()
                html_content += f"""
                        <tr>
                            <td><strong>{col}</strong></td>
                            <td>{stats['mean']:.3f}</td>
                            <td>{stats['std']:.3f}</td>
                            <td>{stats['min']:.3f}</td>
                            <td>{stats['25%']:.3f}</td>
                            <td>{stats['50%']:.3f}</td>
                            <td>{stats['75%']:.3f}</td>
                            <td>{stats['max']:.3f}</td>
                        </tr>
                """
            html_content += """
                    </tbody>
                </table>
            </div>
            """
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            html_content += """
            <div class=\"section\">
                <h2>📝 Categorical Columns Analysis</h2>
            """
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(10)
                html_content += f"""
                <h3>{col}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    html_content += f"""
                        <tr>
                            <td>{value}</td>
                            <td>{count:,}</td>
                            <td>{percentage:.2f}%</td>
                        </tr>
                    """
                html_content += """
                    </tbody>
                </table>
                """
            html_content += "</div>"
        html_content += """
        </body>
        </html>
        """
        return {"report_html": html_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating profile report: {e}")


@app.post("/api/model/train")
async def train_model(request: TrainRequest):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")
        for col in request.feature_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[request.target_column]):
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' must be numerical. Current type: {df[request.target_column].dtype}")
        non_numerical_features = []
        for col in request.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numerical_features.append(col)
        if non_numerical_features:
            raise HTTPException(status_code=400, detail=f"Feature columns must be numerical. Non-numerical columns: {', '.join(non_numerical_features)}")
        X = df[request.feature_columns]
        y = df[request.target_column]
        valid_indices = X.notna().all(axis=1) & y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid data after removing NaN values.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=request.test_size, random_state=request.random_state)
        data_store["X_train"] = X_train
        data_store["X_test"] = X_test
        data_store["y_train"] = y_train
        data_store["y_test"] = y_test
        data_store["target_column"] = request.target_column
        log_action(f"Starting model training: {len(X_train)} train samples, {len(X_test)} test samples")
        if request.model_name.lower() == "linear regression":
            model = LinearRegression(**request.hyperparameters)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not supported.")
        model.fit(X_train, y_train)
        log_action("Model training completed successfully")
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        data_store["trained_model"] = model
        plot_url = "Higher dimension data: 2D plot not possible."
        try:
            if X_test.shape[1] == 1:
                x_vals = X_test.iloc[:, 0].values
                y_true = y_test.values
                y_pred_plot = y_pred
                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                ax.scatter(x_vals, y_true, color="#1f77b4", alpha=0.7, label="Actual")
                sort_idx = np.argsort(x_vals)
                ax.plot(x_vals[sort_idx], y_pred_plot[sort_idx], color="#ff7f0e", linewidth=2, label="Predicted")
                ax.set_xlabel(X_test.columns[0])
                ax.set_ylabel(request.target_column)
                ax.set_title(f"{request.model_name} (R² = {r2:.3f})")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.3)
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                plot_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as plot_err:
            plot_url = f"Plot generation failed: {plot_err}"
        log_action(f"Trained {request.model_name} model with R² score: {r2:.4f}")
        return {"r2_score": float(r2), "model_name": request.model_name, "train_samples": len(X_train), "test_samples": len(X_test), "feature_columns": request.feature_columns, "target_column": request.target_column, "plot_url": plot_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")


@app.get("/api/data/undo")
async def undo_last_action():
    raise HTTPException(status_code=503, detail="Undo feature temporarily disabled during refactor phase.")


@app.get("/api/data/history-length")
async def get_history_length():
    raise HTTPException(status_code=503, detail="Undo feature temporarily disabled during refactor phase.")


@app.get("/api/visualize/scatter")
async def get_scatter_data(
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    sample_size: int = 2000,
    output: str = "json",
):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    if x_col not in df.columns or y_col not in df.columns:
        raise HTTPException(status_code=404, detail="One or both columns not found.")
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        raise HTTPException(status_code=400, detail="Both columns must be numerical.")
    if color_col and color_col not in df.columns:
        raise HTTPException(status_code=404, detail="Color column not found.")
    try:
        columns_to_clean = [x_col, y_col] + ([color_col] if color_col else [])
        clean_df = df[columns_to_clean].dropna()
        if len(clean_df) > sample_size:
            clean_df = clean_df.sample(n=sample_size, random_state=42)
        if output == "json":
            response_data: Dict[str, Any] = {"x": clean_df[x_col].tolist(), "y": clean_df[y_col].tolist(), "x_col": x_col, "y_col": y_col}
            if color_col:
                response_data["color"] = clean_df[color_col].tolist()
                response_data["color_col"] = color_col
                log_action(f"Prepared 2D scatter data: {x_col} vs {y_col} colored by {color_col}")
            else:
                log_action(f"Prepared 2D scatter data: {x_col} vs {y_col}")
            return response_data
        if output == "png":
            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            if color_col is None:
                ax.scatter(clean_df[x_col], clean_df[y_col], s=16, alpha=0.75)
            else:
                if pd.api.types.is_numeric_dtype(clean_df[color_col]):
                    sc = ax.scatter(clean_df[x_col], clean_df[y_col], c=clean_df[color_col], cmap="viridis", s=16, alpha=0.85)
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(color_col)
                else:
                    categories = clean_df[color_col].astype("category")
                    codes = categories.cat.codes
                    sc = ax.scatter(clean_df[x_col], clean_df[y_col], c=codes, cmap="tab10", s=16, alpha=0.85)
                    handles = []
                    labels = []
                    for code, name in enumerate(categories.cat.categories):
                        handles.append(matplotlib.lines.Line2D([], [], linestyle="", marker="o", color=sc.cmap(sc.norm(code)), markersize=6))
                        labels.append(str(name))
                    ax.legend(handles, labels, title=color_col, loc="best", fontsize=8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")
            ax.grid(True, linestyle="--", alpha=0.3)
            buf = BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            log_action(f"Generated 2D scatter image: {x_col} vs {y_col}{' colored by ' + color_col if color_col else ''}")
            return {"image": data_url, "x_col": x_col, "y_col": y_col, "color_col": color_col}
        raise HTTPException(status_code=400, detail="Invalid output type. Use 'json' or 'png'.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scatter plot: {e}")

@app.get("/api/checkpoints/list")
async def list_checkpoints():
    checkpoints = data_store.get("checkpoints", [])
    meta = []
    for cp in checkpoints:
        df = cp.get("df")
        meta.append({
            "id": cp.get("id"),
            "timestamp": cp.get("timestamp"),
            "description": cp.get("description"),
            "rows": int(len(df)) if df is not None else 0,
            "columns": int(len(df.columns)) if df is not None else 0,
        })
    return {"checkpoints": meta}

@app.post("/api/checkpoints/save")
async def save_checkpoint(payload: Dict[str, Any]):
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    description = payload.get("description")
    confirm_eviction = bool(payload.get("confirm_eviction", False))
    checkpoints = data_store.get("checkpoints")
    if len(checkpoints) >= 5 and not confirm_eviction:
        raise HTTPException(status_code=409, detail="Maximum checkpoint limit reached (5 datasets). Confirmation required to evict oldest.")
    try:
        if len(checkpoints) >= 5 and confirm_eviction:
            checkpoints.pop(0)
        import datetime as dt
        cp = {
            "id": f"cp-{int(dt.datetime.now().timestamp()*1000)}",
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "df": df.copy()
        }
        checkpoints.append(cp)
        log_action("Saved dataset checkpoint")
        return {"id": cp["id"], "timestamp": cp["timestamp"], "description": description, "rows": len(df), "columns": len(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving checkpoint: {e}")

@app.get("/api/checkpoints/export")
async def export_checkpoint(id: str):
    checkpoints = data_store.get("checkpoints")
    target = None
    for cp in checkpoints:
        if cp.get("id") == id:
            target = cp
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found.")
    try:
        df = target["df"]
        csv_string = df.to_csv(index=False)
        filename = f"{id}.csv"
        csv_bytes = csv_string.encode("utf-8")
        csv_io = io.BytesIO(csv_bytes)
        return StreamingResponse(iter([csv_io.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting checkpoint: {e}")
