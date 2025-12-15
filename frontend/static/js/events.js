function wireEvents(root) {
  const errorBox = root.querySelector("#errorBox");
  const sidebarItems = root.querySelectorAll(".sidebar-item");
  sidebarItems.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = btn.dataset.target;
      const stage = target.replace("Section", "").replace("upload", "upload").replace("clean", "clean").replace("analyze", "analyze").replace("model", "model").replace("export", "export");
      window.state.setCurrentSection(stage);
      window.render.setActiveSidebar(target);
      window.render.renderStage(stage);
      bindStage(root);
      if (stage === "clean") {
        try {
          const preview = await window.api.getPreview();
          const ms = await window.api.missingSummary();
          const profile = await window.api.getProfile();
          window.appState.profileColumnInfo = profile.column_info || [];
          const columns = preview.columns || [];
          window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, columns);
          window.render.setCleanActive(true);
          window.render.populateMultiSelect(root.querySelector("#fillColumns"), columns);
          window.render.populateMultiSelect(root.querySelector("#dropColumnsSelect"), columns);
          window.render.populateSelect(root.querySelector("#renameSelect"), columns);
          window.render.populateSelect(root.querySelector("#dtypeColumn"), columns);
          window.render.populateMultiSelect(root.querySelector("#labelSelect"), columns);
          window.render.populateMultiSelect(root.querySelector("#onehotSelect"), columns);
          const numCols = profile.numerical_columns || columns;
          window.render.populateMultiSelect(root.querySelector("#scaleSelect"), numCols);
        } catch (e) {
          console.error(e);
          window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), [], []);
          window.render.renderError(errorBox, "Upload a dataset to enable cleaning");
          window.render.setCleanActive(false);
        }
      }
    });
  });

  bindStage(root);
}

function bindStage(root) {
  const errorBox = root.querySelector("#errorBox");
  const fileInput = root.querySelector("#fileInput");
  const uploadBtn = root.querySelector("#uploadBtn");
  const previewContainer = root.querySelector("#preview");
  const metaBox = root.querySelector("#metaBox");

  if (uploadBtn) uploadBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const file = fileInput.files[0];
    if (!file) {
      window.render.renderError(errorBox, "Please select a file first.");
      return;
    }
    const originalText = uploadBtn.textContent;
    window.render.setButtonLoading(uploadBtn, "Uploading…");
    try {
      await window.api.uploadFile(file);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.state.setDatasetMeta({ columns: preview.columns || [], rows: preview.total_rows, columnsCount: preview.total_columns });
      window.render.clearError(errorBox);
      window.render.setActiveSidebar("cleanSection");
      window.render.renderStage("clean");
      bindStage(root);
      window.render.setCleanActive(true);
      try {
        const ms = await window.api.missingSummary();
        const profile = await window.api.getProfile();
        window.appState.profileColumnInfo = profile.column_info || [];
        const columns = preview.columns || [];
        window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, columns);
        window.render.populateMultiSelect(root.querySelector("#fillColumns"), columns);
        window.render.populateMultiSelect(root.querySelector("#dropColumnsSelect"), columns);
        window.render.populateSelect(root.querySelector("#renameSelect"), columns);
        window.render.populateSelect(root.querySelector("#dtypeColumn"), columns);
        window.render.populateMultiSelect(root.querySelector("#labelSelect"), columns);
        window.render.populateMultiSelect(root.querySelector("#onehotSelect"), columns);
        const numCols = profile.numerical_columns || columns;
        window.render.populateMultiSelect(root.querySelector("#scaleSelect"), numCols);
      } catch (e) {
        console.error(e);
      }
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(uploadBtn, originalText);
    }
  });

  const profileBtn = root.querySelector("#profileBtn");
  if (profileBtn) profileBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = profileBtn.textContent;
    window.render.setButtonLoading(profileBtn, "Generating…");
    try {
      const report = await window.api.getProfileReport();
      const blob = new Blob([report.report_html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "data_profile_report.html";
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      a.remove();
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(profileBtn, originalText);
    }
  });

  const exportBtn = root.querySelector("#exportBtn");
  const exportSource = root.querySelector("#exportSource");
  if (exportBtn) exportBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = exportBtn.textContent;
    window.render.setButtonLoading(exportBtn, "Downloading…");
    try {
      const source = exportSource.value;
      let blob;
      if (source === "final") {
        blob = await window.api.exportCSV();
      } else {
        blob = await window.api.exportCheckpoint(source);
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "dataset.csv";
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(url);
      a.remove();
      window.render.clearError(errorBox);
      window.render.renderInfo(root.querySelector("#infoBox"), "Exported dataset successfully.");
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(exportBtn, originalText);
    }
  });

  const dropRowsMissingBtn = root.querySelector("#dropRowsMissingBtn");
  if (dropRowsMissingBtn) dropRowsMissingBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = dropRowsMissingBtn.textContent;
    window.render.setButtonLoading(dropRowsMissingBtn, "Processing…");
    try {
      await window.api.clean([{ type: "drop_rows_with_missing", threshold: 1 }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), "Dropped rows with missing values.");
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(dropRowsMissingBtn, originalText);
    }
  });

  const dropColsMissingBtn = root.querySelector("#dropColsMissingBtn");
  if (dropColsMissingBtn) dropColsMissingBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = dropColsMissingBtn.textContent;
    window.render.setButtonLoading(dropColsMissingBtn, "Processing…");
    try {
      const ms = await window.api.missingSummary();
      const toDrop = ms.filter(x => x.missing_count > 0).map(x => x.column_name);
      if (toDrop.length === 0) {
        window.render.renderError(errorBox, "No columns have missing values.");
        throw new Error("No columns have missing values.");
      }
      await window.api.clean([{ type: "drop_columns", columns: toDrop }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Dropped ${toDrop.length} columns with missing values.`);
      const ms2 = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms2, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(dropColsMissingBtn, originalText);
    }
  });

  const fillMethod = root.querySelector("#fillMethod");
  const fillColumns = root.querySelector("#fillColumns");
  const fillConstant = root.querySelector("#fillConstant");
  const fillMissingBtn = root.querySelector("#fillMissingBtn");
  if (fillMissingBtn) fillMissingBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = fillMissingBtn.textContent;
    window.render.setButtonLoading(fillMissingBtn, "Applying…");
    try {
      const method = fillMethod.value;
      const cols = Array.from(fillColumns.selectedOptions).map(o => o.value);
      const op = { type: "fill_missing", method, columns: cols };
      if (method === "constant") op.value = fillConstant.value;
      await window.api.clean([op]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Filled missing with ${method}.`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(fillMissingBtn, originalText);
    }
  });

  const removeDuplicatesBtn = root.querySelector("#removeDuplicatesBtn");
  if (removeDuplicatesBtn) removeDuplicatesBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = removeDuplicatesBtn.textContent;
    window.render.setButtonLoading(removeDuplicatesBtn, "Processing…");
    try {
      const before = await window.api.getPreview();
      const dupBefore = await fetch(`${API_BASE_URL}/api/data/duplicates_summary`).then(r => r.json());
      await window.api.clean([{ type: "remove_duplicates" }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      const dupAfter = await fetch(`${API_BASE_URL}/api/data/duplicates_summary`).then(r => r.json());
      const removed = (dupBefore.duplicate_count || 0) - (dupAfter.duplicate_count || 0);
      window.render.renderInfo(root.querySelector("#infoBox"), `Removed ${removed} duplicate rows.`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(removeDuplicatesBtn, originalText);
    }
  });

  const dropColumnsInput = root.querySelector("#dropColumnsInput");
  const dropColumnsBtn = root.querySelector("#dropColumnsBtn");
  if (dropColumnsBtn) dropColumnsBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = dropColumnsBtn.textContent;
    window.render.setButtonLoading(dropColumnsBtn, "Dropping…");
    try {
      const cols = Array.from(root.querySelector("#dropColumnsSelect").selectedOptions).map(o => o.value);
      if (cols.length === 0) {
        window.render.renderError(errorBox, "Enter columns to drop.");
        throw new Error("No columns selected");
      }
      await window.api.clean([{ type: "drop_columns", columns: cols }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Dropped ${cols.length} column(s).`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(dropColumnsBtn, originalText);
    }
  });

  const renameSelect = root.querySelector("#renameSelect");
  const renameNew = root.querySelector("#renameNew");
  const renameBtn = root.querySelector("#renameBtn");
  if (renameBtn) renameBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = renameBtn.textContent;
    window.render.setButtonLoading(renameBtn, "Renaming…");
    try {
      if (!renameSelect.value || !renameNew.value) {
        window.render.renderError(errorBox, "Enter old and new column names.");
        throw new Error("Missing rename values");
      }
      await window.api.renameColumn(renameSelect.value, renameNew.value);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Renamed column.`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(renameBtn, originalText);
    }
  });

  const dtypeColumn = root.querySelector("#dtypeColumn");
  const dtypeNew = root.querySelector("#dtypeNew");
  const changeTypeBtn = root.querySelector("#changeTypeBtn");
  if (changeTypeBtn) changeTypeBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = changeTypeBtn.textContent;
    window.render.setButtonLoading(changeTypeBtn, "Converting…");
    try {
      if (!dtypeColumn.value || !dtypeNew.value) {
        window.render.renderError(errorBox, "Enter column and target type.");
        throw new Error("Missing dtype values");
      }
      await window.api.changeType(dtypeColumn.value, dtypeNew.value);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Converted type of ${dtypeColumn.value} to ${dtypeNew.value}.`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(changeTypeBtn, originalText);
    }
  });

  const labelSelect = root.querySelector("#labelSelect");
  const labelEncodeBtn = root.querySelector("#labelEncodeBtn");
  if (labelEncodeBtn) labelEncodeBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = labelEncodeBtn.textContent;
    window.render.setButtonLoading(labelEncodeBtn, "Encoding…");
    try {
      const cols = Array.from(labelSelect.selectedOptions).map(o => o.value);
      if (cols.length === 0) {
        window.render.renderError(errorBox, "Enter columns to encode.");
        throw new Error("No columns selected");
      }
      await window.api.clean([{ type: "label_encode", columns: cols }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Label encoded ${cols.length} column(s).`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(labelEncodeBtn, originalText);
    }
  });

  const onehotSelect = root.querySelector("#onehotSelect");
  const onehotEncodeBtn = root.querySelector("#onehotEncodeBtn");
  if (onehotEncodeBtn) onehotEncodeBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = onehotEncodeBtn.textContent;
    window.render.setButtonLoading(onehotEncodeBtn, "Encoding…");
    try {
      const cols = Array.from(onehotSelect.selectedOptions).map(o => o.value);
      if (cols.length === 0) {
        window.render.renderError(errorBox, "Enter columns to encode.");
        throw new Error("No columns selected");
      }
      await window.api.clean([{ type: "one_hot_encode", columns: cols }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `One-hot encoded ${cols.length} column(s).`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(onehotEncodeBtn, originalText);
    }
  });

  const scaleSelect = root.querySelector("#scaleSelect");
  const scaleMethod = root.querySelector("#scaleMethod");
  const scaleBtn = root.querySelector("#scaleBtn");
  if (scaleBtn) scaleBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = scaleBtn.textContent;
    window.render.setButtonLoading(scaleBtn, "Scaling…");
    try {
      const cols = Array.from(scaleSelect.selectedOptions).map(o => o.value);
      if (cols.length === 0) {
        window.render.renderError(errorBox, "Enter columns to scale.");
        throw new Error("No columns selected");
      }
      await window.api.clean([{ type: "scale", columns: cols, method: scaleMethod.value }]);
      const preview = await window.api.getPreview();
      if (previewContainer) window.render.renderPreview(previewContainer, preview);
      if (metaBox) window.render.renderMetadata(metaBox, preview);
      window.render.renderInfo(root.querySelector("#infoBox"), `Scaled ${cols.length} column(s) with ${scaleMethod.value}.`);
      const ms = await window.api.missingSummary();
      window.render.renderCleanDatasetInfo(root.querySelector("#cleanDatasetInfo"), ms, preview.columns || []);
    } catch (e) {
      console.error(e);
      window.render.renderError(errorBox, e.message);
    } finally {
      window.render.resetButton(scaleBtn, originalText);
    }
  });

  const saveCheckpointBtn = root.querySelector("#saveCheckpointBtn");
  const checkpointDesc = root.querySelector("#checkpointDesc");
  const exportSourceSelect = root.querySelector("#exportSource");
  async function refreshCheckpoints() {
    try {
      const list = await window.api.listCheckpoints();
      window.render.renderCheckpointsSelect(exportSourceSelect, list.checkpoints || []);
    } catch (e) {
      console.error(e);
    }
  }
  if (saveCheckpointBtn) saveCheckpointBtn.addEventListener("click", async () => {
    window.render.clearError(errorBox);
    const originalText = saveCheckpointBtn.textContent;
    window.render.setButtonLoading(saveCheckpointBtn, "Saving…");
    try {
      await window.api.saveCheckpoint(checkpointDesc.value || null, false);
      await refreshCheckpoints();
    } catch (e) {
      console.error(e);
      const msg = String(e.message || "");
      if (msg.includes("Maximum checkpoint limit")) {
        const proceed = window.confirm("Maximum checkpoint limit reached (5 datasets). Remove oldest checkpoint to save the new one?");
        if (proceed) {
          try {
            await window.api.saveCheckpoint(checkpointDesc.value || null, true);
            await refreshCheckpoints();
          } catch (ee) {
            console.error(ee);
            window.render.renderError(errorBox, ee.message);
          }
        }
      } else {
        window.render.renderError(errorBox, e.message);
      }
    } finally {
      window.render.resetButton(saveCheckpointBtn, originalText);
    }
  });
  refreshCheckpoints();
}

window.events = { wireEvents, bindStage };
