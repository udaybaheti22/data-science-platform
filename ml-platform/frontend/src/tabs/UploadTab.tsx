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

const SAMPLE_DATASETS = [
  {
    name: "house_prices",
    label: "House Prices",
    forModel: "For Linear Regression",
    description: "Predict median home value. Practice filling missing values, fixing column types, and removing duplicates.",
    target: "MedianHomeValue",
    rows: 511,
    cols: 14,
    badge: "regression",
  },
  {
    name: "titanic",
    label: "Titanic Survival",
    forModel: "For Decision Tree",
    description: "Predict passenger survival. Practice label encoding, filling missing age values, and dropping irrelevant columns.",
    target: "Survived",
    rows: 423,
    cols: 12,
    badge: "classification",
  },
  {
    name: "iris",
    label: "Iris Flowers",
    forModel: "For KNN",
    description: "Classify flower species. Practice fixing column types, filling missing values, and label encoding.",
    target: "species",
    rows: 154,
    cols: 5,
    badge: "classification",
  },
];

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
  const [sampleLoading, setSampleLoading] = useState<string | null>(null);

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

  async function handleLoadSample(name: string) {
    if (isLoaded) {
      const confirmed = window.confirm(
        "Loading a sample dataset will replace your current dataset and all cleaning progress. Continue?"
      );
      if (!confirmed) return;
      onClearDataset();
    }

    setSampleLoading(name);
    setError(null);
    setPreview(null);

    try {
      const result = await api.loadSampleDataset(name);
      const previewData = await api.getPreview();
      setPreview(previewData);
      onUploadSuccess(result.filename, result.rows, result.columns, result.column_list);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load sample dataset.");
    } finally {
      setSampleLoading(null);
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

      {/* Sample datasets section */}
      <div className="sample-datasets-section">
        <div className="sample-datasets-header">
          <h3>Don't have your own dataset?</h3>
          <p>Try one of these sample datasets — each is pre-loaded with missing values, wrong column types, and duplicates for you to clean and explore.</p>
        </div>

        <div className="sample-cards">
          {SAMPLE_DATASETS.map((ds) => (
            <div key={ds.name} className="sample-card">
              <div className="sample-card-top">
                <span className="sample-for-label">{ds.forModel}</span>
                <span className={`sample-badge sample-badge--${ds.badge}`}>{ds.badge}</span>
              </div>

              <h4 className="sample-card-title">{ds.label}</h4>
              <p className="sample-card-desc">{ds.description}</p>

              <div className="sample-card-meta">
                <span>{ds.rows} rows</span>
                <span>·</span>
                <span>{ds.cols} columns</span>
                <span>·</span>
                <span>Target: <code>{ds.target}</code></span>
              </div>

              <button
                className="btn btn-secondary sample-use-btn"
                onClick={() => handleLoadSample(ds.name)}
                disabled={sampleLoading === ds.name}
              >
                {sampleLoading === ds.name ? "Loading…" : "Use this dataset"}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
