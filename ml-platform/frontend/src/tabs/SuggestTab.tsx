import { useState } from "react";
import { api } from "../api/client";
import type { AISuggestionResponse } from "../types/api";
import "../styles/suggest.css";

interface SuggestTabProps {
  columns: string[];
  aiSuggestions: AISuggestionResponse | null;
  onSuggestionsGenerated: (suggestions: AISuggestionResponse) => void;
}

export default function SuggestTab({ columns, aiSuggestions, onSuggestionsGenerated }: SuggestTabProps) {
  const [targetColumn, setTargetColumn] = useState(aiSuggestions?.target_column ?? "");
  const [taskType, setTaskType] = useState<"regression" | "classification">(
    (aiSuggestions?.task_type as "regression" | "classification") ?? "regression"
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerate(e: React.FormEvent) {
    e.preventDefault();
    if (!targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await api.getAISuggestions({
        target_column: targetColumn,
        task_type: taskType,
      });
      onSuggestionsGenerated(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate suggestions.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="suggest-tab">
      <h2>AI Suggestions</h2>
      <p className="suggest-intro">
        Let AI analyze your dataset and suggest data cleaning operations and optimal model
        hyperparameters based on statistical patterns and correlations.
      </p>

      <form className="suggest-form" onSubmit={handleGenerate}>
        <div className="form-group">
          <label htmlFor="target-col">Target Column</label>
          <select
            id="target-col"
            value={targetColumn}
            onChange={(e) => {
              setTargetColumn(e.target.value);
              setError(null);
            }}
          >
            <option value="">— Select target column —</option>
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <small className="field-hint">
            The column you want to predict (e.g., price, category, outcome).
          </small>
        </div>

        <div className="form-group">
          <label>Task Type</label>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="task-type"
                value="regression"
                checked={taskType === "regression"}
                onChange={(e) => setTaskType(e.target.value as "regression")}
              />
              <span>Regression</span>
              <small>(predicting continuous values like price, age, temperature)</small>
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="task-type"
                value="classification"
                checked={taskType === "classification"}
                onChange={(e) => setTaskType(e.target.value as "classification")}
              />
              <span>Classification</span>
              <small>(predicting categories like spam/not spam, species, disease type)</small>
            </label>
          </div>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading || !targetColumn}>
          {loading ? "Generating suggestions..." : aiSuggestions ? "Regenerate Suggestions" : "Generate AI Suggestions"}
        </button>
      </form>

      {error && <p className="error-msg">⚠ {error}</p>}

      {aiSuggestions && (
        <div className="suggestions-result">
          <h3>AI Analysis Results</h3>

          <div className="summary-box">
            <h4>📊 Summary</h4>
            <p>{aiSuggestions.summary_text}</p>
          </div>

          <div className="suggestions-section">
            <h4>🧹 Data Cleaning Suggestions</h4>
            {aiSuggestions.column_suggestions.length === 0 ? (
              <p className="empty-state">No cleaning operations recommended — dataset looks clean!</p>
            ) : (
              <div className="suggestion-cards">
                {aiSuggestions.column_suggestions.map((sug, idx) => (
                  <div key={idx} className="suggestion-card">
                    <div className="card-header">
                      <strong>{sug.column}</strong>
                      <span className="action-badge">{sug.action.replace(/_/g, " ")}</span>
                    </div>
                    <p className="card-reason">{sug.reason}</p>
                  </div>
                ))}
              </div>
            )}
            <p className="hint-text">
              💡 These suggestions appear as hints in the <strong>Clean</strong> tab when you
              select the corresponding column.
            </p>
          </div>

          <div className="suggestions-section">
            <h4>🤖 Model & Hyperparameter Recommendations</h4>
            <div className="model-card">
              <div className="model-header">
                <strong>Recommended Model:</strong>
                <span className="model-badge">
                  {aiSuggestions.model_suggestions.recommended_model.replace(/_/g, " ")}
                </span>
              </div>
              <div className="hyperparam-list">
                <strong>Suggested Hyperparameters:</strong>
                <ul>
                  {Object.entries(aiSuggestions.model_suggestions.hyperparameters).map(
                    ([key, value]) => (
                      <li key={key}>
                        <code>{key}</code>: <span>{String(value)}</span>
                      </li>
                    )
                  )}
                </ul>
              </div>
            </div>
            <p className="hint-text">
              💡 These hyperparameters appear as hints in the <strong>Model</strong> tab when
              you select this model.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
