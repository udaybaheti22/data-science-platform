interface HyperparamPanelProps {
  modelType: "linear_regression" | "decision_tree" | "knn";
  hyperparams: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}

export default function HyperparamPanel({ modelType, hyperparams, onChange }: HyperparamPanelProps) {
  return (
    <div className="hyperparam-panel">
      <h4>Hyperparameters</h4>

      {/* Shared: test_size */}
      <div className="param-row">
        <label htmlFor="test_size">Test Size</label>
        <select
          id="test_size"
          value={String(hyperparams.test_size ?? "0.2")}
          onChange={(e) => onChange("test_size", parseFloat(e.target.value))}
        >
          {["0.1", "0.2", "0.3", "0.4"].map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
      </div>

      {modelType === "linear_regression" && (
        <div className="param-row">
          <label htmlFor="fit_intercept">Fit Intercept</label>
          <select
            id="fit_intercept"
            value={String(hyperparams.fit_intercept ?? "true")}
            onChange={(e) => onChange("fit_intercept", e.target.value === "true")}
          >
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
      )}

      {modelType === "decision_tree" && (
        <>
          <div className="param-row">
            <label htmlFor="max_depth">Max Depth</label>
            <select
              id="max_depth"
              value={String(hyperparams.max_depth ?? "null")}
              onChange={(e) =>
                onChange("max_depth", e.target.value === "null" ? null : parseInt(e.target.value))
              }
            >
              <option value="null">None (unlimited)</option>
              {[2, 4, 6, 8, 10].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="param-row">
            <label htmlFor="min_samples_split">Min Samples Split</label>
            <select
              id="min_samples_split"
              value={String(hyperparams.min_samples_split ?? "2")}
              onChange={(e) => onChange("min_samples_split", parseInt(e.target.value))}
            >
              {[2, 5, 10].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="param-row">
            <label htmlFor="criterion">Criterion</label>
            <select
              id="criterion"
              value={String(hyperparams.criterion ?? "gini")}
              onChange={(e) => onChange("criterion", e.target.value)}
            >
              <option value="gini">gini (classification)</option>
              <option value="entropy">entropy (classification)</option>
              <option value="squared_error">squared_error (regression)</option>
              <option value="absolute_error">absolute_error (regression)</option>
            </select>
          </div>
        </>
      )}

      {modelType === "knn" && (
        <>
          <div className="param-row">
            <label htmlFor="n_neighbors">Neighbours (k)</label>
            <select
              id="n_neighbors"
              value={String(hyperparams.n_neighbors ?? "5")}
              onChange={(e) => onChange("n_neighbors", parseInt(e.target.value))}
            >
              {[3, 5, 7, 9, 11].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div className="param-row">
            <label htmlFor="weights">Weights</label>
            <select
              id="weights"
              value={String(hyperparams.weights ?? "uniform")}
              onChange={(e) => onChange("weights", e.target.value)}
            >
              <option value="uniform">uniform</option>
              <option value="distance">distance</option>
            </select>
          </div>

          <div className="param-row">
            <label htmlFor="metric">Distance Metric</label>
            <select
              id="metric"
              value={String(hyperparams.metric ?? "euclidean")}
              onChange={(e) => onChange("metric", e.target.value)}
            >
              <option value="euclidean">euclidean</option>
              <option value="manhattan">manhattan</option>
              <option value="minkowski">minkowski</option>
            </select>
          </div>
        </>
      )}
    </div>
  );
}
