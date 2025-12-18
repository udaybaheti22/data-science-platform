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

// Profiling
export async function generateProfileReport() {
  const resp = await fetch(`${API_BASE_URL}/api/data/profile_report`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
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
