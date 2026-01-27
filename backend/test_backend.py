# Simplified backend for testing core functionality
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
import json
import uuid
import os
from typing import Dict, Any
import datetime

# Initialize the FastAPI app
app = FastAPI(title="Test EDA API")

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5500",
    "http://localhost:5501",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Data Storage
data_store = {
    "original_df": None,
    "main_df": None,
    "history": [],
    "logs": []
}

def log_action(description: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_store["logs"].insert(0, {"timestamp": timestamp, "description": description})

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        data_store["original_df"] = df.copy()
        data_store["main_df"] = df.copy()
        
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
    df = data_store.get("original_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    preview_df = df.head(limit)
    
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

@app.get("/api/data/changes_preview")
async def get_changes_preview():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    sample_size = min(30, len(df))
    if sample_size == 0:
        preview_data = []
    else:
        preview_df = df.sample(n=sample_size, random_state=42)
        
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
        "preview_rows": len(preview_data)
    }

@app.get("/api/data/column_types")
async def get_column_types():
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")
    try:
        columns_info = []
        for col in df.columns:
            columns_info.append({
                "name": col,
                "current_type": str(df[col].dtype),
                "non_null_count": int(df[col].count())
            })
        return columns_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting column types: {e}")

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
        
        supported_types = ["int64", "float64", "object", "bool", "datetime64[ns]"]
        if new_type not in supported_types:
            raise HTTPException(status_code=400, detail=f"Unsupported data type '{new_type}'. Supported types: {', '.join(supported_types)}")
        
        current_type = str(df[column_name].dtype)
        if current_type == new_type:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' is already of type '{new_type}'")
        
        # Save current state for undo
        data_store["history"].append(df.copy())
        
        # Attempt to convert the column type
        try:
            if new_type == "datetime64[ns]":
                df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            elif new_type in ["int64", "float64"]:
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype(new_type)
            elif new_type == "bool":
                df[column_name] = df[column_name].astype(str).str.lower().map({
                    'true': True, 'false': False, '1': True, '0': False,
                    'yes': True, 'no': False, 't': True, 'f': False,
                    'y': True, 'n': False
                }).fillna(False).astype(bool)
            else:
                df[column_name] = df[column_name].astype(new_type)
            
            log_action(f"Changed type of '{column_name}' from '{current_type}' to '{new_type}'.")
            
            # Get 30 random rows for changes preview
            sample_size = min(30, len(df))
            if sample_size > 0:
                preview_df = df.sample(n=sample_size, random_state=42)
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
            else:
                preview_data = []
            
            return {
                "message": f"Successfully changed type of '{column_name}' to '{new_type}'",
                "rows": len(df),
                "columns": len(df.columns),
                "history_length": len(data_store["history"]),
                "old_type": current_type,
                "new_type": new_type,
                "changes_preview": {
                    "data": preview_data,
                    "columns": df.columns.tolist(),
                    "total_rows": len(df),
                    "preview_rows": len(preview_data)
                }
            }
            
        except ValueError as e:
            if data_store["history"]:
                data_store["main_df"] = data_store["history"].pop()
            raise HTTPException(status_code=400, detail=f"Conversion failed: {str(e)}")
        except Exception as e:
            if data_store["history"]:
                data_store["main_df"] = data_store["history"].pop()
            raise HTTPException(status_code=400, detail=f"Error converting column type: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error changing column type: {e}")

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
        
        new_column_name = new_column_name.strip()
        if not new_column_name:
            raise HTTPException(status_code=400, detail="New column name cannot be empty or whitespace")
        
        if old_column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{old_column_name}' not found")
        
        if new_column_name in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{new_column_name}' already exists")
        
        if old_column_name == new_column_name:
            raise HTTPException(status_code=400, detail=f"New name must be different from current name")
        
        # Save current state for undo
        data_store["history"].append(df.copy())
        
        # Rename the column
        df.rename(columns={old_column_name: new_column_name}, inplace=True)
        log_action(f"Renamed column '{old_column_name}' to '{new_column_name}'.")
        
        # Get 30 random rows for changes preview
        sample_size = min(30, len(df))
        if sample_size > 0:
            preview_df = df.sample(n=sample_size, random_state=42)
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
        else:
            preview_data = []
        
        return {
            "message": f"Successfully renamed column '{old_column_name}' to '{new_column_name}'",
            "rows": len(df),
            "columns": len(df.columns),
            "history_length": len(data_store["history"]),
            "old_name": old_column_name,
            "new_name": new_column_name,
            "changes_preview": {
                "data": preview_data,
                "columns": df.columns.tolist(),
                "total_rows": len(df),
                "preview_rows": len(preview_data)
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error renaming column: {e}")

@app.get("/api/data/profile_report")
async def get_profile_report():
    """Mock profile report endpoint for testing"""
    df = data_store.get("main_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset found. Please upload a file first.")

    try:
        log_action("Generated mock profile report.")
        
        # Create a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mock Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dataset Profile Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Rows: {len(df)} | Columns: {len(df.columns)}</p>
            </div>
            
            <div class="section">
                <h2>Column Information</h2>
                <table>
                    <tr><th>Column</th><th>Type</th><th>Non-Null Count</th></tr>
        """
        
        for col in df.columns:
            html_content += f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{df[col].count()}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Sample Data</h2>
                <table>
        """
        
        # Add header
        html_content += "<tr>"
        for col in df.columns:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"
        
        # Add sample rows
        for _, row in df.head(5).iterrows():
            html_content += "<tr>"
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    html_content += "<td>NaN</td>"
                else:
                    html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <p><em>This is a mock profile report for testing. In production, this would be generated by ydata-profiling.</em></p>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        os.makedirs("reports", exist_ok=True)
        unique_id = str(uuid.uuid4())[:8]
        report_path = f"reports/profile_{unique_id}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return FileResponse(path=report_path, filename=f"profile_{unique_id}.html", media_type='text/html')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating profile report: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)