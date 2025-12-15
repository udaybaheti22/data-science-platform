document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("root");
  root.innerHTML = `
    <div class="flex min-h-screen app-bg">
      <aside class="w-56 border-r p-4 sticky top-0 self-start sidebar">
        <h1 class="text-xl font-bold mb-4">Data Science Platform</h1>
        <nav class="space-y-2">
          <button class="w-full sidebar-item" data-target="uploadSection">Upload</button>
          <button class="w-full sidebar-item" data-target="cleanSection">Clean</button>
          <button class="w-full sidebar-item" data-target="analyzeSection">Analyze</button>
          <button class="w-full sidebar-item" data-target="modelSection">Model</button>
          <button class="w-full sidebar-item" data-target="exportSection">Export</button>
        </nav>
      </aside>
      <main class="flex-1 p-6 space-y-6">
        <div id="errorBox" class="text-red-600 hidden"></div>
        <div id="infoBox" class="hidden"></div>
        <div id="content"></div>
      </main>
    </div>
  `;
  window.render.renderStage("upload");
  window.render.setActiveSidebar("uploadSection");
  window.events.wireEvents(root);
});
