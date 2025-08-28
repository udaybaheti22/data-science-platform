export const API_BASE_URL = "http://127.0.0.1:8000";

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/api/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function getPreview() {
  const res = await fetch(`${API_BASE_URL}/api/data/preview`);
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function getProfile() {
  const res = await fetch(`${API_BASE_URL}/api/data/profile`);
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function getLogs() {
  const res = await fetch(`${API_BASE_URL}/api/project/logs`);
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function cleanData(body) {
  const res = await fetch(`${API_BASE_URL}/api/data/clean`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function trainModel(body) {
  const res = await fetch(`${API_BASE_URL}/api/model/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function exportCsv(name = 'full') {
  const url = name === 'full' ? `${API_BASE_URL}/api/data/export` : `${API_BASE_URL}/api/data/export?name=${name}`;
  const res = await fetch(url);
  if (!res.ok) throw await res.json();
  return res.blob();
}


