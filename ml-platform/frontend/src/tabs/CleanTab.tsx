import { useState, useEffect } from "react";
import { api } from "../api/client";
import PreviewTable from "../components/PreviewTable";
import type { MissingSummaryItem, PreviewResponse, AISuggestionResponse } from "../types/api";
import "../styles/clean.css";

type Operation =
  | "change_type"
  | "rename_column"
  | "fill_missing"
  | "drop_rows_with_missing"
  | "drop_columns"
  | "remove_duplicates"
  | "one_hot_encode"
  | "label_encode";

interface CleanTabProps {
  columns: string[];
  rowCount: number | null;
  onColumnsChange: (columns: string[], rowCount?: number) => void;
  aiSuggestions: AISuggestionResponse | null;
}

export default function CleanTab({ columns, rowCount, onColumnsChange, aiSuggestions }: CleanTabProps) {
  const [selectedOp, setSelectedOp] = useState<Operation>("fill_missing");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [missingSummary, setMissingSummary] = useState<MissingSummaryItem[]>([]);
  const [duplicateCount, setDuplicateCount] = useState<number | null>(null);

  // Form fields
  const [selectedColumn, setSelectedColumn] = useState("");
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [newColumnName, setNewColumnName] = useState("");
  const [newType, setNewType] = useState("float64");
  const [fillMethod, setFillMethod] = useState("mean");

  useEffect(() => {
    loadSummaries();
    loadPreview();
  }, []);

  async function loadSummaries() {
    try {
      const missing = await api.getMissingSummary();
      setMissingSummary(missing);
      const dupes = await api.getDuplicatesSummary();
      setDuplicateCount(dupes.duplicate_count);
    } catch {
      // non-critical
    }
  }

  async function loadPreview() {
    try {
      const p = await api.getPreview();
      setPreview(p);
    } catch {
      // non-critical
    }
  }

  async function handleApply(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      let result;

      if (selectedOp === "change_type") {
        result = await api.changeType({ column_name: selectedColumn, new_type: newType });
      } else if (selectedOp === "rename_column") {
        result = await api.renameColumn({
          old_column_name: selectedColumn,
          new_column_name: newColumnName,
        });
      } else if (selectedOp === "fill_missing") {
        result = await api.clean({
          operations: [{ type: "fill_missing", columns: [selectedColumn], method: fillMethod }],
        });
      } else if (selectedOp === "drop_rows_with_missing") {
        result = await api.clean({ operations: [{ type: "drop_rows_with_missing" }] });
      } else if (selectedOp === "drop_columns") {
        result = await api.clean({ operations: [{ type: "drop_columns", columns: selectedColumns }] });
      } else if (selectedOp === "remove_duplicates") {
        result = await api.clean({ operations: [{ type: "remove_duplicates" }] });
      } else if (selectedOp === "one_hot_encode") {
        result = await api.clean({ operations: [{ type: "one_hot_encode", columns: selectedColumns }] });
      } else if (selectedOp === "label_encode") {
        result = await api.clean({ operations: [{ type: "label_encode", columns: selectedColumns }] });
      }

      if (result) {
        setSuccessMsg(result.message);
        onColumnsChange(result.column_list, result.rows);
        await loadPreview();
        await loadSummaries();
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Operation failed.");
    } finally {
      setLoading(false);
    }
  }

  const ops: { id: Operation; label: string }[] = [
    { id: "fill_missing", label: "Fill Missing Values" },
    { id: "drop_rows_with_missing", label: "Drop Rows with Missing" },
    { id: "remove_duplicates", label: "Remove Duplicates" },
    { id: "change_type", label: "Change Data Type" },
    { id: "rename_column", label: "Rename Column" },
    { id: "drop_columns", label: "Drop Column(s)" },
    { id: "one_hot_encode", label: "One-Hot Encode" },
    { id: "label_encode", label: "Label Encode" },
  ];

  const needsSingleCol = ["fill_missing", "change_type", "rename_column"].includes(selectedOp);
  const needsMultiCol = ["drop_columns", "one_hot_encode", "label_encode"].includes(selectedOp);

  // AI hint for the currently selected single column
  function getSingleColHint(): { action: string; reason: string } | null {
    if (!aiSuggestions || !selectedColumn) return null;
    return aiSuggestions.column_suggestions.find((s) => s.column === selectedColumn) ?? null;
  }

  // AI hints for multi-select: show hints for all currently selected columns
  function getMultiColHints(): { column: string; action: string; reason: string }[] {
    if (!aiSuggestions) return [];
    const targets = selectedColumns.length > 0 ? selectedColumns : [];
    return aiSuggestions.column_suggestions.filter((s) => targets.includes(s.column));
  }

  const singleColHint = getSingleColHint();
  const multiColHints = getMultiColHints();

  return (
    <div className="clean-tab">
      <h2>Clean Dataset</h2>

      <div className="clean-layout">
        {/* Left: summaries */}
        <aside className="clean-sidebar">
          <div className="summary-card">
            <h4>Missing Values</h4>
            {missingSummary.length === 0 ? (
              <p className="empty-state">None detected.</p>
            ) : (
              <table className="summary-table">
                <thead><tr><th>Column</th><th>Missing</th></tr></thead>
                <tbody>
                  {missingSummary.map((item) => (
                    <tr key={item.column_name}>
                      <td>{item.column_name}</td>
                      <td className={item.missing_count > 0 ? "warn" : ""}>{item.missing_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="summary-card">
            <h4>Duplicate Rows</h4>
            <p className={duplicateCount ? "warn" : ""}>
              {duplicateCount === null ? "—" : `${duplicateCount} duplicate(s)`}
            </p>
          </div>

          <div className="summary-card">
            <h4>Current Shape</h4>
            <p>{rowCount ?? "—"} rows × {columns.length} columns</p>
          </div>
        </aside>

        {/* Right: operation form */}
        <section className="clean-main">
          {/* AI suggestions overview — shown whenever suggestions exist */}
          {aiSuggestions && aiSuggestions.column_suggestions.length > 0 && (
            <div className="ai-suggestions-panel">
              <p className="ai-panel-title">💡 AI Suggestions</p>
              <ul className="ai-suggestion-list">
                {aiSuggestions.column_suggestions.map((s, i) => (
                  <li key={i} className="ai-suggestion-item">
                    <span className="ai-col-name">{s.column}</span>
                    <span className="ai-action-tag">{s.action.replace(/_/g, " ")}</span>
                    <span className="ai-reason">{s.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <form className="clean-form" onSubmit={handleApply}>
            <div className="form-group">
              <label htmlFor="op-select">Operation</label>
              <select
                id="op-select"
                value={selectedOp}
                onChange={(e) => {
                  setSelectedOp(e.target.value as Operation);
                  setError(null);
                  setSuccessMsg(null);
                  setSelectedColumn(columns[0] ?? "");
                  setSelectedColumns([]);
                }}
              >
                {ops.map((op) => (
                  <option key={op.id} value={op.id}>{op.label}</option>
                ))}
              </select>
            </div>

            {needsSingleCol && (
              <div className="form-group">
                <label htmlFor="single-col">Column</label>
                <select
                  id="single-col"
                  value={selectedColumn}
                  onChange={(e) => setSelectedColumn(e.target.value)}
                >
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                {singleColHint && (
                  <div className="ai-hint">
                    <span className="ai-hint-action">{singleColHint.action.replace(/_/g, " ")}</span>
                    <span className="ai-hint-reason">{singleColHint.reason}</span>
                  </div>
                )}
              </div>
            )}

            {needsMultiCol && (
              <div className="form-group">
                <label htmlFor="multi-col">Columns (hold Ctrl/Cmd for multi-select)</label>
                <select
                  id="multi-col"
                  multiple
                  value={selectedColumns}
                  onChange={(e) =>
                    setSelectedColumns(Array.from(e.target.selectedOptions, (o) => o.value))
                  }
                  size={Math.min(8, columns.length)}
                >
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                {multiColHints.length > 0 && (
                  <div className="ai-hint-multi">
                    {multiColHints.map((h, i) => (
                      <div key={i} className="ai-hint">
                        <span className="ai-hint-col">{h.column}:</span>
                        <span className="ai-hint-action">{h.action.replace(/_/g, " ")}</span>
                        <span className="ai-hint-reason">{h.reason}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {selectedOp === "fill_missing" && (
              <div className="form-group">
                <label htmlFor="fill-method">Fill Method</label>
                <select
                  id="fill-method"
                  value={fillMethod}
                  onChange={(e) => setFillMethod(e.target.value)}
                >
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="mode">Mode</option>
                </select>
              </div>
            )}

            {selectedOp === "change_type" && (
              <div className="form-group">
                <label htmlFor="new-type">Target Type</label>
                <select
                  id="new-type"
                  value={newType}
                  onChange={(e) => setNewType(e.target.value)}
                >
                  <option value="float64">float64</option>
                  <option value="int64">int64</option>
                  <option value="object">object (string)</option>
                  <option value="bool">bool</option>
                </select>
              </div>
            )}

            {selectedOp === "rename_column" && (
              <div className="form-group">
                <label htmlFor="new-col-name">New Column Name</label>
                <input
                  id="new-col-name"
                  type="text"
                  value={newColumnName}
                  onChange={(e) => setNewColumnName(e.target.value)}
                  placeholder="Enter new name…"
                />
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                loading ||
                (needsSingleCol && !selectedColumn) ||
                (needsMultiCol && selectedColumns.length === 0) ||
                (selectedOp === "rename_column" && !newColumnName.trim())
              }
            >
              {loading ? "Applying…" : "Apply"}
            </button>
          </form>

          {error && <p className="error-msg">⚠ {error}</p>}
          {successMsg && <p className="success-msg">✓ {successMsg}</p>}
        </section>
      </div>

      {/* Data preview */}
      {preview && (
        <div className="clean-preview">
          <h3>Current Data Preview</h3>
          <PreviewTable columns={preview.columns} data={preview.data} />
        </div>
      )}
    </div>
  );
}
