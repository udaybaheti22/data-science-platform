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

window.api = { uploadFile, getPreview };
