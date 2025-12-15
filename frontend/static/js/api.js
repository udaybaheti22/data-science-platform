const API_BASE_URL = "http://127.0.0.1:8000";

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE_URL}/api/upload`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function getPreview(limit = 50) {
  const resp = await fetch(`${API_BASE_URL}/api/data/preview?limit=${limit}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function getProfileReport() {
  const resp = await fetch(`${API_BASE_URL}/api/data/profile_report`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function getProfile() {
  const resp = await fetch(`${API_BASE_URL}/api/data/profile`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function exportCSV() {
  const resp = await fetch(`${API_BASE_URL}/api/data/export?name=full`);
  if (!resp.ok) throw new Error(await resp.text());
  const blob = await resp.blob();
  return blob;
}

async function clean(operations) {
  const resp = await fetch(`${API_BASE_URL}/api/data/clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ operations })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function renameColumn(oldName, newName) {
  const resp = await fetch(`${API_BASE_URL}/api/data/rename_column`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_column_name: oldName, new_column_name: newName })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function changeType(column, newType) {
  const resp = await fetch(`${API_BASE_URL}/api/data/change_type`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column_name: column, new_type: newType })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function missingSummary() {
  const resp = await fetch(`${API_BASE_URL}/api/data/missing_summary`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function listCheckpoints() {
  const resp = await fetch(`${API_BASE_URL}/api/checkpoints/list`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function saveCheckpoint(description, confirmEviction = false) {
  const resp = await fetch(`${API_BASE_URL}/api/checkpoints/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description, confirm_eviction: confirmEviction })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function exportCheckpoint(id) {
  const resp = await fetch(`${API_BASE_URL}/api/checkpoints/export?id=${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.blob();
}

window.api = { uploadFile, getPreview, getProfileReport, getProfile, exportCSV, clean, renameColumn, changeType, missingSummary, listCheckpoints, saveCheckpoint, exportCheckpoint };
