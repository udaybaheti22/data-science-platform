import { useState } from "react";
import { api } from "../api/client";
import HyperparamPanel from "../components/HyperparamPanel";
import type { TrainResponse, TrainMetricsRegression, TrainMetricsClassification } from "../types/api";
import "../styles/model.css";

type ModelType = "linear_regression" | "decision_tree" | "knn";

interface ModelTabProps {
  columns: string[];
}

const DEFAULT_HYPERPARAMS: Record<ModelType, Record<string, unknown>> = {
  linear_regression: { fit_intercept: true, test_size: 0.2 },
  decision_tree: { max_depth: null, min_samples_split: 2, criterion: "gini", test_size: 0.2 },
  knn: { n_neighbors: 5, weights: "uniform", metric: "euclidean", test_size: 0.2 },
};

export default function ModelTab({ columns }: ModelTabProps) {
  const [modelType, setModelType] = useState<ModelType>("linear_regression");
  const [hyperparams, setHyperparams] = useState<Record<string, unknown>>(
    DEFAULT_HYPERPARAMS.linear_regression
  );
  const [targetColumn, setTargetColumn] = useState("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrainResponse | null>(null);

  function handleModelTypeChange(newType: ModelType) {
    setModelType(newType);
    setHyperparams({ ...DEFAULT_HYPERPARAMS[newType] });
    setResult(null);
    setError(null);
  }

  function handleHyperparamChange(key: string, value: unknown) {
    setHyperparams((prev) => ({ ...prev, [key]: value }));
  }

  async function handleTrain(e: React.FormEvent) {
    e.preventDefault();
    if (featureColumns.length === 0) {
      setError("Please select at least 1 feature column.");
      return;
    }
    if (!targetColumn) {
      setError("Please select a target column.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const base = {
        target_column: targetColumn,
        feature_columns: featureColumns,
        test_size: hyperparams.test_size as number,
        random_state: 42,
      };

      let res: TrainResponse;
      if (modelType === "linear_regression") {
        res = await api.trainLinearRegression({
          ...base,
          fit_intercept: hyperparams.fit_intercept as boolean,
        });
      } else if (modelType === "decision_tree") {
        res = await api.trainDecisionTree({
          ...base,
          max_depth: hyperparams.max_depth as number | null,
          min_samples_split: hyperparams.min_samples_split as number,
          criterion: hyperparams.criterion as string,
        });
      } else {
        res = await api.trainKNN({
          ...base,
          n_neighbors: hyperparams.n_neighbors as number,
          weights: hyperparams.weights as string,
          metric: hyperparams.metric as string,
        });
      }
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Training failed.");
    } finally {
      setLoading(false);
    }
  }

  function isRegression(metrics: TrainResponse["metrics"]): metrics is TrainMetricsRegression {
    return "mse" in metrics;
  }

  return (
    <div className="model-tab">
      <h2>Train a Model</h2>

      <div className="model-note">
        ℹ️ Only <strong>numeric columns</strong> can be used as features or target.
        Use the <strong>Clean tab</strong> to label-encode or one-hot encode text columns first.
      </div>

      <form className="model-form" onSubmit={handleTrain}>
        <div className="form-group">
          <label htmlFor="model-type">Model Type</label>
          <select
            id="model-type"
            value={modelType}
            onChange={(e) => handleModelTypeChange(e.target.value as ModelType)}
          >
            <option value="linear_regression">Linear Regression</option>
            <option value="decision_tree">Decision Tree</option>
            <option value="knn">K-Nearest Neighbours (KNN)</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="target-col">Target Column</label>
          <select
            id="target-col"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
          >
            <option value="">— Select target —</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="feature-cols">
            Feature Columns <small>(hold Ctrl/Cmd to select multiple)</small>
          </label>
          <select
            id="feature-cols"
            multiple
            value={featureColumns}
            onChange={(e) =>
              setFeatureColumns(Array.from(e.target.selectedOptions, (o) => o.value))
            }
            size={Math.min(8, columns.length)}
          >
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          {featureColumns.length === 0 && (
            <small className="field-hint">Select at least 1 feature.</small>
          )}
          {featureColumns.length > 2 && (
            <small className="field-hint muted">
              2D visualisation requires ≤ 2 features — only metrics will be shown.
            </small>
          )}
        </div>

        <HyperparamPanel
          modelType={modelType}
          hyperparams={hyperparams}
          onChange={handleHyperparamChange}
        />

        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || featureColumns.length === 0 || !targetColumn}
        >
          {loading ? "Training…" : "Train Model"}
        </button>
      </form>

      {error && (
        <div className="error-box">
          <p className="error-msg">⚠ {error}</p>
          {error.includes("Non-numeric") && (
            <p className="error-hint">
              Go to <strong>Clean tab</strong> → select the text column → apply{" "}
              <em>Label Encode</em> or <em>One-Hot Encode</em>, then come back.
            </p>
          )}
        </div>
      )}

      {result && (
        <div className="model-results">
          <h3>Results — {result.model_name.replace(/_/g, " ").toUpperCase()}</h3>
          <p className="task-type-badge">{result.task_type}</p>

          {result.dropped_rows > 0 && (
            <p className="warn-msg">
              ℹ {result.dropped_rows} row(s) with NaN in features were dropped before training.
            </p>
          )}

          <table className="metrics-table">
            <thead>
              <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
              {isRegression(result.metrics) ? (
                <>
                  <tr><td>MSE</td><td>{result.metrics.mse.toFixed(4)}</td></tr>
                  <tr><td>RMSE</td><td>{result.metrics.rmse.toFixed(4)}</td></tr>
                  <tr><td>MAE</td><td>{result.metrics.mae.toFixed(4)}</td></tr>
                  <tr><td>R²</td><td>{result.metrics.r2.toFixed(4)}</td></tr>
                </>
              ) : (
                <>
                  <tr><td>Accuracy</td><td>{((result.metrics as TrainMetricsClassification).accuracy * 100).toFixed(2)}%</td></tr>
                  <tr><td>Precision</td><td>{((result.metrics as TrainMetricsClassification).precision * 100).toFixed(2)}%</td></tr>
                  <tr><td>Recall</td><td>{((result.metrics as TrainMetricsClassification).recall * 100).toFixed(2)}%</td></tr>
                  <tr><td>F1 Score</td><td>{((result.metrics as TrainMetricsClassification).f1 * 100).toFixed(2)}%</td></tr>
                </>
              )}
            </tbody>
          </table>

          {result.viz_base64 ? (
            <div className="viz-section">
              <h4>2D Visualisation</h4>
              <img
                src={`data:image/png;base64,${result.viz_base64}`}
                alt="Model 2D visualisation"
                className="viz-img"
              />
            </div>
          ) : featureColumns.length > 2 ? (
            <p className="muted-msg">2D visualisation requires ≤ 2 features.</p>
          ) : null}

          {result.tree_base64 && (
            <div className="tree-section">
              <h4>Decision Tree Structure</h4>
              <div className="tree-scroll">
                <img
                  src={`data:image/png;base64,${result.tree_base64}`}
                  alt="Decision Tree diagram"
                  className="tree-img"
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
