document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("root");
  root.innerHTML = `
    <div class="p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h1 class="text-xl font-bold">Data Science Platform</h1>
        <button id="themeToggle" class="btn">Toggle Theme</button>
      </div>
      <div class="card p-4">
        <div class="flex gap-2 items-center">
          <input id="fileInput" type="file" accept=".csv" class="border p-2 rounded" />
          <button id="uploadBtn" class="btn btn-primary">Upload & Preview</button>
        </div>
      </div>
      <div class="card p-4">
        <h2 class="font-semibold mb-2">Preview</h2>
        <div id="preview" class="table-container"></div>
      </div>
    </div>
  `;
  const themeToggle = document.getElementById("themeToggle");
  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("dark");
  });
  window.events.wireEvents(root);
});
