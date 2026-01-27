import {
  uploadDataset,
  getPreview,
  getChangesPreview,
  getMissingValues,
  getColumnTypes,
  changeColumnType,
  renameColumn,
  PROFILE_REPORT_URL,
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
  renderDatatypeTable,
  renderChangesPreview,
} from "./render.js";

function wireSidebar() {
  const items = document.querySelectorAll(".sidebar-item");
  console.log(`🔗 Found ${items.length} sidebar items`);
  
  items.forEach((item) => {
    item.addEventListener("click", () => {
      const sectionId = item.getAttribute("data-section");
      console.log(`🖱️  Sidebar clicked: ${sectionId}`);
      
      showSection(sectionId);
      setActiveSidebar(sectionId);
      clearAlerts();

      if (sectionId === "section-clean") {
        console.log("🧹 Loading Clean section...");
        loadCleanSection();
      } else if (sectionId === "section-data") {
        console.log("📊 Loading Data section...");
        loadDataSection();
      }
      // section-preview, section-analyze,
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
    
    // Preload data for Clean and Data sections so they're ready when user navigates
    try {
      // Preload column types for Data section
      const columns = await getColumnTypes();
      console.log("Preloaded column types for Data section:", columns.length, "columns");
      
      // Preload missing values for Clean section  
      const missingValues = await getMissingValues();
      console.log("Preloaded missing values for Clean section:", missingValues.length, "columns");
      
    } catch (preloadErr) {
      console.warn("Failed to preload section data:", preloadErr);
      // Don't show error to user, sections will load when navigated to
    }
    
    showInfo("Dataset uploaded and preview loaded. Clean and Data sections are ready.");
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
  const chooseFileButton = document.getElementById("choose-file-button");
  const uploadInput = document.getElementById("upload-input");
  const selectedFilename = document.getElementById("selected-filename");
  const uploadButton = document.getElementById("upload-button");
  
  if (chooseFileButton && uploadInput && selectedFilename && uploadButton) {
    // Wire choose file button to trigger file input
    chooseFileButton.addEventListener("click", () => {
      uploadInput.click();
    });
    
    // Handle file selection
    uploadInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        selectedFilename.textContent = file.name;
        selectedFilename.style.color = "#374151"; // Darker color when file selected
        selectedFilename.style.fontStyle = "normal";
        uploadButton.disabled = false;
      } else {
        selectedFilename.textContent = "No file selected";
        selectedFilename.style.color = "#6b7280"; // Gray color
        selectedFilename.style.fontStyle = "italic";
        uploadButton.disabled = true;
      }
    });
    
    // Handle upload button click
    uploadButton.addEventListener("click", () => {
      handleUploadClick();
    });
  }
}

async function loadDataSection() {
  console.log("Loading Data Section...");
  const container = document.getElementById("column-types-wrapper");
  if (!container) return;

  try {
    const columns = await getColumnTypes();
    renderDatatypeTable(container, columns);
  } catch (err) {
    console.error(err);
    // Handle missing dataset gracefully
    if (err.message && (err.message.includes("No dataset") || err.message.includes("404"))) {
      container.innerHTML = `
        <div class="placeholder-description" style="text-align: center; padding: 20px;">
          <p>No dataset found.</p>
          <button class="btn btn-primary" onclick="document.querySelector('[data-section=\\'section-preview\\']').click()">Go to Upload</button>
        </div>
      `;
    } else {
      showError("Unable to load column types. " + err.message);
    }
  }
}

function wireDataSection() {
  const container = document.getElementById("column-types-wrapper");
  if (!container) return;
  
  // Use event delegation for dynamic elements
  container.addEventListener("click", async (e) => {
    if (e.target.dataset.action === "save-type") {
      await handleTypeChange(e.target);
    } else if (e.target.dataset.action === "save-rename") {
      await handleColumnRename(e.target);
    }
  });
}

async function handleTypeChange(btn) {
  const colName = btn.dataset.columnName;
  
  // Find the row and its components
  const row = btn.closest("tr");
  const select = row.querySelector("select");
  const currentTypeCell = row.querySelector("td[data-current-type]");
  const errorMsg = row.querySelector(".inline-error-message[data-error-type='type']");
  
  if (!select || !currentTypeCell || !errorMsg) {
    console.error("Could not find required elements for column:", colName);
    return;
  }
  
  const newType = select.value;
  const currentType = currentTypeCell.dataset.currentType;

  // Clear any existing error message
  errorMsg.style.display = "none";
  errorMsg.textContent = "";

  const originalText = btn.textContent;
  btn.textContent = "Saving...";
  btn.disabled = true;

  try {
    const result = await changeColumnType(colName, newType);
    
    // Update the current type display immediately
    currentTypeCell.textContent = newType;
    currentTypeCell.dataset.currentType = newType;
    
    // Reset the dropdown to the new current type
    select.value = newType;
    
    // Disable the save button since there's no change now
    btn.disabled = true;
    btn.className = "btn btn-secondary";
    btn.textContent = "Change Type";
    
    // Update changes preview if it exists
    updateChangesPreview(result.changes_preview);
    
    console.log(`Successfully converted column '${colName}' to ${newType}`);
    
  } catch (err) {
    console.error("Error converting column type:", err);
    
    // Show inline error message for this specific row
    errorMsg.textContent = err.message || "Failed to change column type";
    errorMsg.style.display = "block";
    
    // Reset button state
    btn.textContent = originalText;
    btn.disabled = false;
    btn.className = "btn btn-primary";
  }
}

async function handleColumnRename(btn) {
  const colName = btn.dataset.columnName;
  
  // Find the row and its components
  const row = btn.closest("tr");
  const nameInput = row.querySelector("input[type='text']");
  const currentNameCell = row.querySelector("td[data-current-name]");
  const errorMsg = row.querySelector(".inline-error-message[data-error-type='rename']");
  
  if (!nameInput || !currentNameCell || !errorMsg) {
    console.error("Could not find required elements for column:", colName);
    return;
  }
  
  const newName = nameInput.value.trim();
  const currentName = currentNameCell.dataset.currentName;

  // Clear any existing error message
  errorMsg.style.display = "none";
  errorMsg.textContent = "";

  const originalText = btn.textContent;
  btn.textContent = "Saving...";
  btn.disabled = true;

  try {
    const result = await renameColumn(currentName, newName);
    
    // Update all references to this column in the row
    currentNameCell.textContent = newName;
    currentNameCell.dataset.currentName = newName;
    
    // Update the input value
    nameInput.value = newName;
    
    // Update data attributes
    row.dataset.columnName = newName;
    nameInput.dataset.columnName = newName;
    btn.dataset.columnName = newName;
    
    // Update the static column name cell
    const staticNameCell = row.querySelector("td:first-child");
    if (staticNameCell) {
      staticNameCell.textContent = newName;
    }
    
    // Disable the save button since there's no change now
    btn.disabled = true;
    btn.className = "btn btn-secondary";
    btn.textContent = "Rename";
    
    // Update changes preview if it exists
    updateChangesPreview(result.changes_preview);
    
    console.log(`Successfully renamed column '${currentName}' to '${newName}'`);
    
  } catch (err) {
    console.error("Error renaming column:", err);
    
    // Show inline error message for this specific row
    errorMsg.textContent = err.message || "Failed to rename column";
    errorMsg.style.display = "block";
    
    // Reset button state
    btn.textContent = originalText;
    btn.disabled = false;
    btn.className = "btn btn-primary";
  }
}

function updateChangesPreview(previewData) {
  const previewContainer = document.getElementById("changes-preview-wrapper");
  if (previewContainer && previewData) {
    renderChangesPreview(previewContainer, previewData);
  }
}

function wireAnalyzeSection() {
  const btn = document.getElementById("generate-analyze-report");
  if (!btn) {
    console.error("Generate analyze report button not found!");
    return;
  }
  
  console.log("Wiring analyze section button...");
  
  btn.addEventListener("click", (e) => {
    console.log("Generate profile button clicked!");
    e.preventDefault();
    
    // Clear any existing alerts
    clearAlerts();
    
    try {
      // Open the profile report in a new tab
      console.log("Opening profile report URL:", PROFILE_REPORT_URL);
      window.open(PROFILE_REPORT_URL, '_blank');
      showInfo("Opening profile report in new tab...");
    } catch (error) {
      console.error("Error opening profile report:", error);
      showError("Failed to open profile report: " + error.message);
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
  console.log("🔌 Wiring sidebar events...");
  wireSidebar();
  
  console.log("📤 Wiring upload events...");
  wireUploadSection();
  
  console.log("📊 Wiring data section events...");
  wireDataSection();
  
  console.log("📈 Wiring analyze section events...");
  wireAnalyzeSection();
  
  console.log("📦 Wiring export section events...");
  wireExportSection();
  
  console.log("✅ All events wired successfully");
}

// Entry point
window.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 DOM Content Loaded - Initializing app...");
  
  // Initial state: show preview section
  showSection("section-preview");
  setActiveSidebar("section-preview");
  
  console.log("📡 Wiring events...");
  wireEvents();
  
  console.log("✅ App initialization complete");
});
