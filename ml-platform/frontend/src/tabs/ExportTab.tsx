import { useState } from "react";
import { api } from "../api/client";
import "../styles/export.css";

export default function ExportTab() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  async function handleExport() {
    setLoading(true);
    setError(null);
    setSuccess(false);
    try {
      const blob = await api.exportDataset();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "dataset.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setSuccess(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="export-tab">
      <h2>Export Dataset</h2>
      <p className="section-desc">
        Download the current cleaned dataset as a CSV file.
      </p>

      <button
        className="btn btn-primary"
        onClick={handleExport}
        disabled={loading}
      >
        {loading ? "Exporting…" : "⬇ Export Dataset"}
      </button>

      {error && <p className="error-msg">⚠ {error}</p>}
      {success && <p className="success-msg">✓ Download started.</p>}
    </div>
  );
}
