import { useState } from "react";
import { api } from "../api/client";
import PreviewTable from "../components/PreviewTable";
import type { PreviewResponse } from "../types/api";
import "../styles/upload.css";

interface UploadTabProps {
  isLoaded: boolean;
  filename: string | null;
  rowCount: number | null;
  colCount: number | null;
  onUploadSuccess: (filename: string, rows: number, cols: number, columns: string[]) => void;
  onClearDataset: () => void;
  onColumnsChange: (columns: string[], rowCount?: number) => void;
}

export default function UploadTab({
  isLoaded,
  filename,
  rowCount,
  colCount,
  onUploadSuccess,
  onClearDataset,
  onColumnsChange,
}: UploadTabProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [assignHeaderLoading, setAssignHeaderLoading] = useState(false);

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedFile) return;

    if (isLoaded) {
      const confirmed = window.confirm(
        "Uploading a new file will replace your current dataset and all cleaning progress. Continue?"
      );
      if (!confirmed) return;
      onClearDataset();
    }

    setLoading(true);
    setError(null);
    setPreview(null);

    try {
      const result = await api.upload(selectedFile);
      const previewData = await api.getPreview();
      setPreview(previewData);
      onUploadSuccess(result.filename, result.rows, result.columns, previewData.columns);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAssignHeader() {
    setAssignHeaderLoading(true);
    setError(null);
    try {
      const result = await api.assignHeader();
      const previewData = await api.getPreview();
      setPreview(previewData);
      onColumnsChange(result.columns, result.rows);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to assign header.");
    } finally {
      setAssignHeaderLoading(false);
    }
  }

  return (
    <div className="upload-tab">
      <h2>Upload Dataset</h2>

      <form className="upload-form" onSubmit={handleUpload}>
        <div className="file-input-row">
          <input
            type="file"
            accept=".csv"
            id="csv-file-input"
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
          />
          <label htmlFor="csv-file-input" className="file-label">
            {selectedFile ? selectedFile.name : "Choose a CSV file…"}
          </label>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!selectedFile || loading}
          >
            {loading ? "Uploading…" : "Upload"}
          </button>
        </div>
      </form>

      {error && <p className="error-msg">⚠ {error}</p>}

      {isLoaded && preview && (
        <div className="upload-preview">
          <div className="dataset-summary">
            <span>📊 <strong>{filename}</strong></span>
            <span>{rowCount} rows</span>
            <span>{colCount} columns</span>
          </div>

          <PreviewTable columns={preview.columns} data={preview.data} />

          <div className="assign-header-section">
            <button
              className="btn btn-secondary"
              onClick={handleAssignHeader}
              disabled={assignHeaderLoading}
            >
              {assignHeaderLoading ? "Assigning…" : "Use first row as column header"}
            </button>
            <small>Use this if your CSV has no header row.</small>
          </div>
        </div>
      )}

      {!isLoaded && (
        <div className="upload-placeholder">
          <p>No dataset loaded. Upload a CSV file to get started.</p>
        </div>
      )}
    </div>
  );
}
