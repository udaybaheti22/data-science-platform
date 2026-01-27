const API_BASE_URL = "http://127.0.0.1:8000";

// Preview & upload
export async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);
  // Backend expects /api/upload (existing implementation)
  const resp = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getPreview(limit = 50) {
  const resp = await fetch(`${API_BASE_URL}/api/data/preview?limit=${limit}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getChangesPreview() {
  const resp = await fetch(`${API_BASE_URL}/api/data/changes_preview`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

// Column metadata / cleaning
export async function getColumnTypes() {
  const resp = await fetch(`${API_BASE_URL}/api/data/column_types`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function cleanColumns(payload) {
  const resp = await fetch(`${API_BASE_URL}/api/data/clean_columns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function getMissingValues() {
  const resp = await fetch(`${API_BASE_URL}/api/data/missing_summary`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function changeColumnType(columnName, newType) {
  const resp = await fetch(`${API_BASE_URL}/api/data/change_type`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column_name: columnName, new_type: newType }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

export async function renameColumn(oldName, newName) {
  const resp = await fetch(`${API_BASE_URL}/api/data/rename_column`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_column_name: oldName, new_column_name: newName }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

// Profiling
export const PROFILE_REPORT_URL = `${API_BASE_URL}/api/data/profile_report`;

export async function generateProfileReport() {
  // Deprecated: use window.open(PROFILE_REPORT_URL)
  const resp = await fetch(PROFILE_REPORT_URL);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.blob(); // Changed from json to blob as it returns file now
}

// Plotting
export async function generatePlot(plotType) {
  const resp = await fetch(`${API_BASE_URL}/api/plot?type=${encodeURIComponent(plotType)}`);
  if (!resp.ok) throw new Error(await resp.text());
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

// Logs
export async function getLogs() {
  const resp = await fetch(`${API_BASE_URL}/api/logs/recent`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

// Export
export async function exportDataset(format = "csv") {
  const resp = await fetch(`${API_BASE_URL}/api/data/export?format=${encodeURIComponent(format)}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.blob();
}
