import {
  uploadDataset,
  getPreview,
  getMissingValues,
  generateProfileReport,
  exportDataset,
} from "./api.js";

import {
  showSection,
  setActiveSidebar,
  showInfo,
  showError,
  clearAlerts,
} from "./ui.js";

import {
  renderPreviewTable,
  renderPreviewMeta,
  renderMissingValues,
} from "./render.js";

function wireSidebar() {
  const items = document.querySelectorAll(".sidebar-item");
  items.forEach((item) => {
    item.addEventListener("click", () => {
      const sectionId = item.getAttribute("data-section");
      showSection(sectionId);
      setActiveSidebar(sectionId);
      clearAlerts();

      if (sectionId === "section-clean") {
        loadCleanSection();
      }
      // section-preview, section-profile, section-data, section-analyze,
      // section-build-model, section-logs, section-export currently do not
      // trigger any backend calls on navigation to avoid hitting
      // non-existent APIs.
    });
  });
}

async function handleUploadClick() {
  clearAlerts();
  const input = document.getElementById("upload-input");
  const button = document.getElementById("upload-button");
  const tableContainer = document.getElementById("preview-table-wrapper");
  const metaContainer = document.getElementById("preview-meta");

  if (!input || !button) return;
  const file = input.files && input.files[0];
  if (!file) {
    showError("Please select a file first.");
    return;
  }

  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "Uploading…";

  try {
    await uploadDataset(file);
    const preview = await getPreview(50);
    renderPreviewTable(tableContainer, preview);
    renderPreviewMeta(metaContainer, preview);
    showInfo("Dataset uploaded and preview loaded.");
  } catch (err) {
    console.error(err);
    showError(String(err));
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function loadCleanSection() {
  try {
    const missingWrapper = document.getElementById("missing-values-wrapper");
    if (!missingWrapper) return;
    const summary = await getMissingValues();
    renderMissingValues(missingWrapper, summary.rows || summary || []);
  } catch (err) {
    console.error(err);
    showError("Unable to load missing value summary.");
  }
}

function wireUploadSection() {
  const button = document.getElementById("upload-button");
  if (button) {
    button.addEventListener("click", () => {
      handleUploadClick();
    });
  }
}

function wireProfileSection() {
  const btn = document.getElementById("generate-profile");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    clearAlerts();
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
      const report = await generateProfileReport();
      const blob = new Blob([report.report_html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "data_profile_report.html";
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      a.remove();
      showInfo("Profile report generated.");
    } catch (err) {
      console.error(err);
      showError("Failed to generate profile report.");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
}

function wireAnalyzeSection() {
  // Reuse profile generation in Analyze section if present (no /api/plot backend)
  const btn = document.getElementById("generate-plot");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    clearAlerts();
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Generating…";
    try {
      const report = await generateProfileReport();
      const blob = new Blob([report.report_html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "data_profile_report.html";
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      a.remove();
      showInfo("Profile report generated.");
    } catch (err) {
      console.error(err);
      showError("Failed to generate profile report.");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
}

function wireExportSection() {
  const btn = document.getElementById("export-dataset");
  const formatSelect = document.getElementById("export-format");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    clearAlerts();
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Exporting…";
    try {
      const fmt = formatSelect ? formatSelect.value : "csv";
      const blob = await exportDataset(fmt);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fmt === "csv" ? "dataset.csv" : "dataset.parquet";
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      a.remove();
      showInfo("Dataset exported.");
    } catch (err) {
      console.error(err);
      showError("Failed to export dataset.");
    } finally {
      btn.disabled = false;
      btn.textContent = originalText;
    }
  });
}

export function wireEvents() {
  wireSidebar();
  wireUploadSection();
  wireProfileSection();
  wireAnalyzeSection();
  wireExportSection();
}

// Entry point
window.addEventListener("DOMContentLoaded", () => {
  // Initial state: show preview section
  showSection("section-preview");
  setActiveSidebar("section-preview");
  wireEvents();
});
