# main.py - The Python Backend using FastAPI

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import json
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pydantic import BaseModel
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# import plotly.utils
import json

# Initialize the FastAPI app
app = FastAPI(
    title="Exploratory Data Analysis API",
    description="An API for performing EDA on uploaded datasets."
)

# --- CORS Middleware ---
# This allows your frontend (running on a different port) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500", # Common for VS Code Live Server
    "http://127.0.0.1:5501", # VS Code Live Server alternative port
    "http://127.0.0.1:3000",
    "http://0.0.0.0:8000",   # Add this for any origin
    "*",                     # Allow all origins for development
    "null" # Allows opening the HTML file directly
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import datetime

# --- In-Memory Data Storage ---
# In a real application, you would use a more robust solution like Redis or a database
# to handle multiple users and sessions. For this example, we'll use a simple global
# dictionary to store the dataframe for the current session.
data_store = {
    "main_df": None,
    "project_states": {},  # Store project states with their modified datasets
    "history": [], # A stack to hold previous versions of the dataframe for undo
    "logs": []     # A list to store log entries
}

# --- Logging Helper ---
def log_action(description: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_store["logs"].insert(0, {"timestamp": timestamp, "description": description}) # Insert at beginning for reverse-chrono order


# --- API Endpoints ---

# Pydantic model for model training requests
class TrainRequest(BaseModel):
    target_column: str
    feature_columns: List[str]
    test_size: float
    random_state: int
    model_name: str
    hyperparameters: Dict[str, Any]

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Handles the CSV file upload, reads it into a pandas DataFrame,
    and stores it in our simple in-memory store.
    """
    print(f"Received upload request for file: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        print(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        # Read the file content
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        # Use io.BytesIO to read the byte string into pandas
        df = pd.read_csv(io.BytesIO(contents))
        
        # Store the dataframe in our global store
        data_store["main_df"] = df
        
        # Log the upload action
        log_action(f"Uploaded dataset: {file.filename} ({len(df)} rows, {len(df.columns)} columns)")
        
        print(f"Successfully uploaded: {file.filename} ({len(df)} rows, {len(df.columns)} columns)")
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns)
        }
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


@app.get("/api/data/preview")
async def get_data_preview(limit: int = 50):
    """
    Returns a preview of the dataset for display in a table format.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    # Get preview data (first 'limit' rows)
    preview_df = df.head(limit)
    
    # Convert NaN values to "NaN" string for JSON serialization
    preview_data = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = "NaN"
            else:
                row_dict[col] = value
        preview_data.append(row_dict)
    
    return {
        "data": preview_data,
        "columns": df.columns.tolist(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "preview_rows": len(preview_df)
    }


@app.get("/api/data/duplicates_summary")
async def get_duplicates_summary():
    """
    Returns a summary of duplicate rows in the dataset.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        # Calculate total number of duplicate rows
        duplicate_count = df.duplicated().sum()
        
        return {
            "duplicate_count": int(duplicate_count)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating duplicates: {e}")


@app.post("/api/data/clean")
async def clean_dataset(cleaning_operations: Dict[str, Any]):
    """
    Applies cleaning operations to the dataset and returns the modified dataset.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        # Save current state for undo
        data_store["history"].append(df.copy())
        
        # Create a copy of the dataframe for modifications
        modified_df = df.copy()
        
        # Apply cleaning operations
        operations = cleaning_operations.get("operations", [])
        
        for operation in operations:
            op_type = operation.get("type")
            
            if op_type == "drop_columns":
                columns_to_drop = operation.get("columns", [])
                modified_df = modified_df.drop(columns=columns_to_drop, errors='ignore')
                log_action(f"Dropped columns: {', '.join(columns_to_drop)}")
            
            elif op_type == "drop_rows_with_missing":
                threshold = operation.get("threshold", 0.5)
                modified_df = modified_df.dropna(thresh=len(modified_df.columns) * threshold)
                log_action(f"Dropped rows with missing values (threshold: {threshold})")
            
            elif op_type == "fill_missing":
                method = operation.get("method", "mean")
                columns = operation.get("columns", [])
                
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
                
                log_action(f"Filled missing values in columns: {', '.join(columns)} using {method} method")
            
            elif op_type == "remove_duplicates":
                limit = operation.get("limit", None)
                
                if limit is None:
                    # Remove all duplicates
                    original_count = len(modified_df)
                    modified_df = modified_df.drop_duplicates()
                    removed_count = original_count - len(modified_df)
                    log_action(f"Removed {removed_count} duplicate rows.")
                else:
                    # Remove only the specified number of duplicates
                    duplicate_indices = modified_df[modified_df.duplicated()].index
                    if len(duplicate_indices) > 0:
                        # Take only the first 'limit' duplicate indices
                        indices_to_drop = duplicate_indices[:limit]
                        modified_df = modified_df.drop(indices_to_drop)
                        log_action(f"Removed {len(indices_to_drop)} duplicate rows (limited).")
                    else:
                        log_action("No duplicate rows found to remove.")
            
            elif op_type == "one_hot_encode":
                columns_to_encode = operation.get("columns", [])
                
                if not columns_to_encode:
                    raise HTTPException(status_code=400, detail="No columns specified for one-hot encoding")
                
                # Verify all columns exist
                missing_columns = [col for col in columns_to_encode if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                
                # Perform one-hot encoding
                encoded_df = pd.get_dummies(modified_df, columns=columns_to_encode)
                modified_df = encoded_df
                
                log_action(f"One-Hot Encoded columns: {', '.join(columns_to_encode)}")
            
            elif op_type == "label_encode":
                columns_to_encode = operation.get("columns", [])
                
                if not columns_to_encode:
                    raise HTTPException(status_code=400, detail="No columns specified for label encoding")
                
                # Verify all columns exist
                missing_columns = [col for col in columns_to_encode if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                
                # Perform label encoding for each column
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
                
                # Verify all columns exist
                missing_columns = [col for col in columns_to_scale if col not in modified_df.columns]
                if missing_columns:
                    raise HTTPException(status_code=400, detail=f"Columns not found: {', '.join(missing_columns)}")
                
                # Verify columns are numerical
                non_numerical_columns = [col for col in columns_to_scale if not pd.api.types.is_numeric_dtype(modified_df[col])]
                if non_numerical_columns:
                    raise HTTPException(status_code=400, detail=f"Non-numerical columns cannot be scaled: {', '.join(non_numerical_columns)}")
                
                # Apply scaling based on method
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
        
        # Update the main dataframe with the cleaned version
        data_store["main_df"] = modified_df
        
        return {
            "message": "Dataset cleaned successfully",
            "rows": len(modified_df),
            "columns": len(modified_df.columns),
            "history_length": len(data_store["history"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {e}")


@app.post("/api/data/change_type")
async def change_column_type(column_data: Dict[str, str]):
    """
    Changes the data type of a specific column.
    """
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
        
        # Save current state for undo
        data_store["history"].append(df.copy())
        
        # Attempt to convert the column type
        try:
            df[column_name] = df[column_name].astype(new_type)
            log_action(f"Changed type of '{column_name}' to '{new_type}'.")
            
            return {
                "message": f"Successfully changed type of '{column_name}' to '{new_type}'",
                "rows": len(df),
                "columns": len(df.columns),
                "history_length": len(data_store["history"])
            }
            
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
    """
    Renames a specific column in the dataset.
    """
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
        
        # Save current state for undo
        data_store["history"].append(df.copy())
        
        # Rename the column
        df.rename(columns={old_column_name: new_column_name}, inplace=True)
        log_action(f"Renamed column '{old_column_name}' to '{new_column_name}'.")
        
        return {
            "message": f"Successfully renamed column '{old_column_name}' to '{new_column_name}'",
            "rows": len(df),
            "columns": len(df.columns),
            "history_length": len(data_store["history"])
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming column: {e}")


@app.post("/api/project/save")
async def save_project_state(project_data: Dict[str, Any]):
    """
    Saves the current state of a project, including any modifications to the dataset.
    """
    try:
        project_id = project_data.get("project_id")
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
        
        # Store the project state
        data_store["project_states"][project_id] = project_data
        
        return {"message": "Project state saved successfully", "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving project state: {e}")


@app.get("/api/project/logs")
async def get_project_logs():
    """
    Returns the action logs for the current project.
    """
    return {
        "logs": data_store["logs"]
    }


@app.get("/api/project/{project_id}")
async def get_project_state(project_id: str):
    """
    Retrieves the saved state of a project.
    """
    project_state = data_store["project_states"].get(project_id)
    if not project_state:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return project_state


@app.get("/api/data/missing_summary")
async def get_missing_summary():
    """
    Returns a summary of missing values for each column in the dataset.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        # Calculate missing values for each column
        missing_counts = df.isnull().sum()
        
        # Convert to list of objects
        missing_summary = []
        for column_name, missing_count in missing_counts.items():
            missing_summary.append({
                "column_name": column_name,
                "missing_count": int(missing_count)
            })
        
        return missing_summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating missing values: {e}")


@app.get("/api/data/export")
async def export_dataset(name: str = 'full'):
    """
    Exports the dataset as a CSV file for download.
    Supports exporting full dataset, training set, or test set.
    """
    try:
        if name == 'train':
            # Export training set
            X_train = data_store.get('X_train')
            y_train = data_store.get('y_train')
            if X_train is None or y_train is None:
                raise HTTPException(status_code=404, detail="Training set not found. Please train a model first.")
            
            # Combine features and target
            train_df = X_train.copy()
            train_df[data_store.get('target_column', 'target')] = y_train
            
            csv_string = train_df.to_csv(index=False)
            filename = "train_dataset.csv"
            
        elif name == 'test':
            # Export test set
            X_test = data_store.get('X_test')
            y_test = data_store.get('y_test')
            if X_test is None or y_test is None:
                raise HTTPException(status_code=404, detail="Test set not found. Please train a model first.")
            
            # Combine features and target
            test_df = X_test.copy()
            test_df[data_store.get('target_column', 'target')] = y_test
            
            csv_string = test_df.to_csv(index=False)
            filename = "test_dataset.csv"
            
        else:
            # Export full dataset
            df = data_store.get("main_df")
            if df is None:
                raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
            
            csv_string = df.to_csv(index=False)
            filename = "dataset.csv"
        
        # Create a streaming response
        csv_bytes = csv_string.encode('utf-8')
        csv_io = io.BytesIO(csv_bytes)
        
        return StreamingResponse(
            iter([csv_io.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {e}")


@app.get("/api/data/profile")
async def get_data_profile():
    """
    Generates a comprehensive profile of the stored dataset.
    This includes column info, descriptive stats, and a data preview.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    # 1. Get Column Information (Name, Type, Non-Null Count)
    info_df = pd.DataFrame({
        'Non-Null Count': df.notna().sum(),
        'Dtype': df.dtypes.astype(str)
    }).reset_index().rename(columns={'index': 'Column'})
    
    # 2. Get Descriptive Statistics for numerical columns
    desc_stats = df.describe().round(3).reset_index().rename(columns={'index': 'Statistic'})

    # 3. Get Value Counts for categorical (object) columns
    value_counts = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        counts = df[col].value_counts().head(5) # Get top 5 for brevity
        value_counts[col] = counts.to_dict()

    # 4. Get a preview of the data (first 5 rows)
    data_head = []
    for _, row in df.head().iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = "NaN"
            else:
                row_dict[col] = value
        data_head.append(row_dict)
    
    # 5. Get a list of numerical columns for the frontend to use
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    return {
        "data_head": data_head,
        "column_info": info_df.to_dict(orient='records'),
        "value_counts": value_counts,
        "numerical_columns": numerical_cols
    }


@app.get("/api/data/profile_report")
async def get_profile_report():
    """
    Generates a comprehensive automated data profiling report.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        # Log the action
        log_action("Generated comprehensive data profiling report.")
        
        # Create a comprehensive HTML report
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
            <div class="header">
                <h1>📊 Data Profile Report</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📈 Dataset Overview</h2>
                <div class="metric">
                    <h3>Total Rows</h3>
                    <div class="value">{len(df):,}</div>
                </div>
                <div class="metric">
                    <h3>Total Columns</h3>
                    <div class="value">{len(df.columns)}</div>
                </div>
                <div class="metric">
                    <h3>Memory Usage</h3>
                    <div class="value">{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</div>
                </div>
                <div class="metric">
                    <h3>Missing Values</h3>
                    <div class="value">{df.isnull().sum().sum():,}</div>
                </div>
            </div>
            
            <div class="section">
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
        
        # Add column information
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
        
        # Add numerical columns statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            html_content += f"""
            <div class="section">
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
        
        # Add categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            html_content += f"""
            <div class="section">
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
        
        # Close the HTML
        html_content += """
        </body>
        </html>
        """
        
        return {"report_html": html_content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating profile report: {e}")


@app.post("/api/model/train")
async def train_model(request: TrainRequest):
    """
    Trains a machine learning model using the specified parameters.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    
    try:
        # Validate columns exist
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found.")
        
        for col in request.feature_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Feature column '{col}' not found.")
        
        # Validate data types - all columns must be numerical for Linear Regression
        if not pd.api.types.is_numeric_dtype(df[request.target_column]):
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' must be numerical. Current type: {df[request.target_column].dtype}"
            )
        
        non_numerical_features = []
        for col in request.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numerical_features.append(col)
        
        if non_numerical_features:
            raise HTTPException(
                status_code=400,
                detail=f"Feature columns must be numerical. Non-numerical columns: {', '.join(non_numerical_features)}"
            )
        
        # Prepare features and target
        X = df[request.feature_columns]
        y = df[request.target_column]
        
        # Drop rows with NaN values in any of the selected columns
        valid_indices = X.notna().all(axis=1) & y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="No valid data after removing NaN values.")
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=request.test_size, 
            random_state=request.random_state
        )
        
        # Store the split datasets for later export
        data_store['X_train'] = X_train
        data_store['X_test'] = X_test
        data_store['y_train'] = y_train
        data_store['y_test'] = y_test
        data_store['target_column'] = request.target_column  # Store for export functionality
        
        # Log training details
        log_action(f"Starting model training: {len(X_train)} train samples, {len(X_test)} test samples")
        
        # Instantiate and train the model
        if request.model_name.lower() == "linear regression":
            model = LinearRegression(**request.hyperparameters)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not supported.")
        
        # Train the model
        model.fit(X_train, y_train)
        log_action(f"Model training completed successfully")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate R-squared score
        r2 = r2_score(y_test, y_pred)
        
        # Store the trained model
        data_store['trained_model'] = model
        
        # Generate plot if 1D feature
        plot_url = "Higher dimension data: 2D plot not possible."
        try:
            if X_test.shape[1] == 1:
                x_vals = X_test.iloc[:, 0].values
                y_true = y_test.values
                y_pred_plot = y_pred
                
                fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
                ax.scatter(x_vals, y_true, color='#1f77b4', alpha=0.7, label='Actual')
                sort_idx = np.argsort(x_vals)
                ax.plot(x_vals[sort_idx], y_pred_plot[sort_idx], color='#ff7f0e', linewidth=2, label='Predicted')
                ax.set_xlabel(X_test.columns[0])
                ax.set_ylabel(request.target_column)
                ax.set_title(f"{request.model_name} (R² = {r2:.3f})")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)
                
                buf = BytesIO()
                plt.tight_layout()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                plot_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as plot_err:
            plot_url = f"Plot generation failed: {plot_err}"
        
        # Log the action
        log_action(f"Trained {request.model_name} model with R² score: {r2:.4f}")
        
        return {
            "r2_score": float(r2),
            "model_name": request.model_name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_columns": request.feature_columns,
            "target_column": request.target_column,
            "plot_url": plot_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")


@app.get("/api/data/undo")
async def undo_last_action():
    """
    Undoes the last action by restoring the previous dataframe state.
    """
    if not data_store["history"]:
        raise HTTPException(status_code=400, detail="No actions to undo.")
    
    try:
        # Pop the last dataframe from history
        previous_df = data_store["history"].pop()
        
        # Set it as the current dataframe
        data_store["main_df"] = previous_df
        
        # Log the undo action
        log_action("Performed Undo")
        
        return {
            "message": "Undo successful",
            "rows": len(previous_df),
            "columns": len(previous_df.columns),
            "history_length": len(data_store["history"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing undo: {e}")


@app.get("/api/data/history-length")
async def get_history_length():
    """
    Returns the current length of the history stack for undo functionality.
    """
    return {
        "history_length": len(data_store["history"])
    }



@app.get("/api/visualize/scatter")
async def get_scatter_data(
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    sample_size: int = 2000,
    output: str = "json"  # "json" or "png"
):
    """
    Generates a 2D scatter plot dataset or image.
    - output=json: returns arrays and column names
    - output=png: returns base64 PNG data URL
    """
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
            response_data: Dict[str, Any] = {
                "x": clean_df[x_col].tolist(),
                "y": clean_df[y_col].tolist(),
                "x_col": x_col,
                "y_col": y_col
            }
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
                # If color column is numeric, use a continuous colormap; otherwise use categorical mapping
                if pd.api.types.is_numeric_dtype(clean_df[color_col]):
                    sc = ax.scatter(clean_df[x_col], clean_df[y_col], c=clean_df[color_col], cmap="viridis", s=16, alpha=0.85)
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(color_col)
                else:
                    categories = clean_df[color_col].astype("category")
                    codes = categories.cat.codes
                    sc = ax.scatter(clean_df[x_col], clean_df[y_col], c=codes, cmap="tab10", s=16, alpha=0.85)
                    # Create legend mapping
                    handles = []
                    labels = []
                    for code, name in enumerate(categories.cat.categories):
                        handles.append(matplotlib.lines.Line2D([], [], linestyle='', marker='o', color=sc.cmap(sc.norm(code)), markersize=6))
                        labels.append(str(name))
                    ax.legend(handles, labels, title=color_col, loc='best', fontsize=8)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")
            ax.grid(True, linestyle='--', alpha=0.3)

            buf = BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format='png')
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


# @app.get("/api/visualize/scatter3d")
# async def get_scatter3d_data(x_col: str, y_col: str, z_col: str, sample_size: int = 2000):
#     """
#     Generates data required for plotting a 3D scatter plot.
#     """
#     df = data_store.get("main_df")
#         raise HTTPException(status_code=404, detail="No dataset found.")
#     
#     if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
#         raise HTTPException(status_code=404, detail="One or more columns not found.")
#         
#     if not all(pd.api.types.is_numeric_dtype(df[col]) for col in [x_col, y_col, z_col]):
#         raise HTTPException(status_code=400, detail="All three columns must be numerical.")
# 
#     try:
#         # Drop rows with NaN values in any of the three columns
#         clean_df = df[[x_col, y_col, z_col]].dropna()
#         
#         # Sample data if dataset is large
#         if len(clean_df) > sample_size:
#             clean_df = clean_df.sample(n=sample_size, random_state=42)
#         
#         log_action(f"Generated 3D scatter plot for columns: {x_col}, {y_col}, {z_col}")
#         
#         return {
#             "x": clean_df[x_col].tolist(),
#             "y": clean_df[y_col].tolist(),
#             "z_col": z_col
#         }
#         
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating 3D scatter plot: {e}")


# @app.get("/api/visualize/correlation")
# async def get_correlation_data():
#     """
#     Generates correlation matrix data for numerical columns.
#     """
#     df = data_store.get("main_df")
#     if df is None:
#         raise HTTPException(status_code=404, detail="No dataset found.")
# 
#     try:
#         # Get only numerical columns
#         numerical_df = df.select_dtypes(include=[np.number])
# 
#         # Calculate correlation matrix
#         corr_matrix = numerical_df.corr()
#         
#         log_action("Generated correlation matrix for numerical columns")
#         
#         return {
#             "correlation_matrix": corr_matrix.round(3).to_dict(),
#             "columns": numerical_df.columns.tolist()
#         }
#         
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating correlation matrix: {e}")


# To run this app:
# 1. Save the code as main.py
# 2. Open your terminal in the same directory
# 3. Run the command: uvicorn main:app --reload