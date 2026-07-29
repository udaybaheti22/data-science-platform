import { useState } from "react";
import { api } from "../api/client";
import "../styles/analyze.css";

interface AnalyzeTabProps {
  corrImg: string | null;
  reportUrl: string | null;
  onCorrImgChange: (img: string | null) => void;
  onReportUrlChange: (url: string | null) => void;
}

export default function AnalyzeTab({ corrImg, reportUrl, onCorrImgChange, onReportUrlChange }: AnalyzeTabProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [corrLoading, setCorrLoading] = useState(false);
  const [corrError, setCorrError] = useState<string | null>(null);

  async function handleGenerateReport() {
    if (isGenerating) return;
    setIsGenerating(true);
    setReportError(null);
    try {
      const { report_url } = await api.getProfileReport();
      onReportUrlChange(report_url);
    } catch (err: unknown) {
      setReportError(err instanceof Error ? err.message : "Failed to generate report.");
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleCorrelationMatrix() {
    setCorrLoading(true);
    setCorrError(null);
    try {
      const { img_base64 } = await api.getCorrelationMatrix();
      onCorrImgChange(img_base64);
    } catch (err: unknown) {
      setCorrError(err instanceof Error ? err.message : "Failed to generate correlation matrix.");
    } finally {
      setCorrLoading(false);
    }
  }

  const apiBase = import.meta.env.VITE_API_URL as string;

  return (
    <div className="analyze-tab">
      <h2>Analyze Dataset</h2>

      {/* Correlation Matrix Section */}
      <section className="analyze-section">
        <h3>Correlation Matrix</h3>
        <p className="section-desc">
          Pearson correlation heatmap for all numeric columns.
        </p>
        <button
          className="btn btn-secondary"
          onClick={handleCorrelationMatrix}
          disabled={corrLoading}
        >
          {corrLoading ? "Generating…" : corrImg ? "Refresh Correlation Matrix" : "Show Correlation Matrix"}
        </button>

        {corrError && <p className="error-msg">⚠ {corrError}</p>}

        {corrImg && (
          <div className="corr-matrix-wrapper">
            <img
              src={`data:image/png;base64,${corrImg}`}
              alt="Pearson Correlation Matrix Heatmap"
              className="corr-matrix-img"
            />
          </div>
        )}
      </section>

      {/* Profile Report Section */}
      <section className="analyze-section">
        <h3>Full Profile Report</h3>
        <p className="section-desc">
          Detailed statistical profile including distributions, missing values, and correlations.
          Generation may take a minute for large datasets.
        </p>
        <button
          className="btn btn-primary"
          onClick={handleGenerateReport}
          disabled={isGenerating}
        >
          {isGenerating ? "Generating report…" : reportUrl ? "Refresh Profile Report" : "Generate Profile Report"}
        </button>

        {isGenerating && (
          <div className="loading-indicator">
            <span className="spinner" aria-label="Loading" /> Generating — this may take a moment…
          </div>
        )}

        {reportError && <p className="error-msg">⚠ {reportError}</p>}

        {reportUrl && !isGenerating && (
          <div className="report-frame-wrapper">
            <iframe
              src={`${apiBase}${reportUrl}`}
              title="Profile Report"
              sandbox="allow-scripts allow-same-origin"
              className="report-frame"
            />
          </div>
        )}
      </section>
    </div>
  );
}
