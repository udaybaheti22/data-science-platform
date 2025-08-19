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
import matplotlib.pyplot as plt
import seaborn as sns
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

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Handles the CSV file upload, reads it into a pandas DataFrame,
    and stores it in our simple in-memory store.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        # Read the file content
        contents = await file.read()
        # Use io.BytesIO to read the byte string into pandas
        df = pd.read_csv(io.BytesIO(contents))
        
        # Store the dataframe in our global store
        data_store["main_df"] = df
        
        # Log the upload action
        log_action(f"Uploaded dataset: {file.filename} ({len(df)} rows, {len(df.columns)} columns)")
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns)
        }
    except Exception as e:
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
async def export_dataset():
    """
    Exports the current dataset as a CSV file for download.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        # Convert dataframe to CSV string
        csv_string = df.to_csv(index=False)
        
        # Create a streaming response
        csv_bytes = csv_string.encode('utf-8')
        csv_io = io.BytesIO(csv_bytes)
        
        return StreamingResponse(
            iter([csv_io.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=dataset.csv"}
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


@app.get("/api/visualize/histogram/{column_name}")
async def get_histogram_data(column_name: str):
    """
    Generates data required for plotting a histogram for a specific numerical column.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    
    if column_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found.")
        
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise HTTPException(status_code=400, detail=f"Column '{column_name}' is not numerical.")

    # Generate histogram data using NumPy
    # We drop NaNs to avoid errors during calculation
    counts, bin_edges = np.histogram(df[column_name].dropna(), bins=20)

    log_action(f"Generated histogram for column: {column_name}")
    
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist()
    }


@app.get("/api/visualize/barchart")
async def get_barchart_data(column: str, sort_by: str = "frequency"):
    """
    Generates data required for plotting a bar chart for a categorical column.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
        
    if pd.api.types.is_numeric_dtype(df[column]):
        raise HTTPException(status_code=400, detail=f"Column '{column}' is numerical. Use histogram instead.")

    try:
        # Get value counts
        value_counts = df[column].value_counts()
        
        # Sort by frequency if requested
        if sort_by == "frequency":
            value_counts = value_counts.sort_values(ascending=False)
        
        # Take top 20 values for better visualization
        top_values = value_counts.head(20)
        
        log_action(f"Generated bar chart for column: {column}")
        
        return {
            "labels": top_values.index.tolist(),
            "values": top_values.values.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating bar chart: {e}")


@app.get("/api/visualize/scatter")
async def get_scatter_data(x_col: str, y_col: str, color_col: str = None, sample_size: int = 2000):
    """
    Generates data required for plotting a 2D scatter plot with optional color coding.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    
    if x_col not in df.columns or y_col not in df.columns:
        raise HTTPException(status_code=404, detail="One or both columns not found.")
        
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        raise HTTPException(status_code=400, detail="Both columns must be numerical.")
    
    # Validate color column if provided
    if color_col and color_col not in df.columns:
        raise HTTPException(status_code=404, detail="Color column not found.")

    try:
        # Prepare columns for cleaning
        columns_to_clean = [x_col, y_col]
        if color_col:
            columns_to_clean.append(color_col)
        
        # Drop rows with NaN values in any of the required columns
        clean_df = df[columns_to_clean].dropna()
        
        # Sample data if dataset is large
        if len(clean_df) > sample_size:
            clean_df = clean_df.sample(n=sample_size, random_state=42)
        
        # Prepare response data
        response_data = {
            "x": clean_df[x_col].tolist(),
            "y": clean_df[y_col].tolist(),
            "x_col": x_col,
            "y_col": y_col
        }
        
        # Add color data if color column is specified
        if color_col:
            response_data["color"] = clean_df[color_col].tolist()
            response_data["color_col"] = color_col
            log_action(f"Generated 2D scatter plot for columns: {x_col} vs {y_col} with color coding: {color_col}")
        else:
            log_action(f"Generated 2D scatter plot for columns: {x_col} vs {y_col}")
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scatter plot: {e}")


@app.get("/api/visualize/scatter3d")
async def get_scatter3d_data(x_col: str, y_col: str, z_col: str, sample_size: int = 2000):
    """
    Generates data required for plotting a 3D scatter plot.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")
    
    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        raise HTTPException(status_code=404, detail="One or more columns not found.")
        
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in [x_col, y_col, z_col]):
        raise HTTPException(status_code=400, detail="All three columns must be numerical.")

    try:
        # Drop rows with NaN values in any of the three columns
        clean_df = df[[x_col, y_col, z_col]].dropna()
        
        # Sample data if dataset is large
        if len(clean_df) > sample_size:
            clean_df = clean_df.sample(n=sample_size, random_state=42)
        
        log_action(f"Generated 3D scatter plot for columns: {x_col}, {y_col}, {z_col}")
        
        return {
            "x": clean_df[x_col].tolist(),
            "y": clean_df[y_col].tolist(),
            "z": clean_df[z_col].tolist(),
            "x_col": x_col,
            "y_col": y_col,
            "z_col": z_col
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating 3D scatter plot: {e}")


@app.get("/api/visualize/correlation")
async def get_correlation_data():
    """
    Generates correlation matrix data for numerical columns.
    """
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found.")

    try:
        # Get only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numerical columns for correlation analysis.")
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        log_action("Generated correlation matrix for numerical columns")
        
        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "columns": numerical_df.columns.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating correlation matrix: {e}")


# To run this app:
# 1. Save the code as main.py
# 2. Open your terminal in the same directory
# 3. Run the command: uvicorn main:app --reload