// Rendering helpers for the vanilla dashboard UI

export function renderPreviewTable(container, preview) {
  if (!container) return;
  const { data = [], columns = [] } = preview || {};
  container.innerHTML = "";

  if (!columns.length) {
    const empty = document.createElement("p");
    empty.className = "placeholder-description";
    empty.textContent = "Upload a dataset to see a preview table.";
    container.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "ds-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  data.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = row[col];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

export function renderPreviewMeta(container, meta) {
  if (!container || !meta) return;
  container.textContent = `Rows: ${meta.total_rows ?? "-"} • Columns: ${meta.total_columns ?? "-"}`;
}

export function renderDatatypeTable(container, columns) {
  if (!container) return;
  container.innerHTML = "";
  const cols = columns || [];
  if (!cols.length) {
    const p = document.createElement("p");
    p.className = "placeholder-description";
    p.textContent = "Upload a dataset to configure column types.";
    container.appendChild(p);
    return;
  }

  const table = document.createElement("table");
  table.className = "ds-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Column Name", "Current Name", "New Name", "Current Data Type", "Target Data Type", "Action"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  cols.forEach((col) => {
    const tr = document.createElement("tr");
    tr.dataset.columnName = col.name;

    // Column Name (static)
    const nameTd = document.createElement("td");
    nameTd.textContent = col.name;

    // Current Name (read-only)
    const currentNameTd = document.createElement("td");
    currentNameTd.textContent = col.name;
    currentNameTd.dataset.currentName = col.name;

    // New Name (text input)
    const newNameTd = document.createElement("td");
    const nameWrapper = document.createElement("div");
    nameWrapper.style.display = "flex";
    nameWrapper.style.flexDirection = "column";
    nameWrapper.style.gap = "5px";

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "field-input field-input-inline";
    nameInput.value = col.name;
    nameInput.dataset.columnName = col.name;
    nameInput.placeholder = "Enter new name";

    const nameErrorMsg = document.createElement("div");
    nameErrorMsg.className = "inline-error-message";
    nameErrorMsg.style.color = "#dc3545";
    nameErrorMsg.style.fontSize = "12px";
    nameErrorMsg.style.display = "none";
    nameErrorMsg.dataset.columnName = col.name;
    nameErrorMsg.dataset.errorType = "rename";

    nameWrapper.appendChild(nameInput);
    nameWrapper.appendChild(nameErrorMsg);
    newNameTd.appendChild(nameWrapper);

    // Current Data Type (read-only)
    const typeTd = document.createElement("td");
    typeTd.textContent = col.current_type;
    typeTd.dataset.currentType = col.current_type;

    // Target Data Type (dropdown)
    const targetTypeTd = document.createElement("td");
    const typeWrapper = document.createElement("div");
    typeWrapper.style.display = "flex";
    typeWrapper.style.flexDirection = "column";
    typeWrapper.style.gap = "5px";

    const select = document.createElement("select");
    select.className = "field-input field-input-inline";
    select.dataset.columnName = col.name;
    ["int64", "float64", "object", "bool", "datetime64[ns]"].forEach((opt) => {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (opt === col.current_type) option.selected = true;
      select.appendChild(option);
    });

    const typeErrorMsg = document.createElement("div");
    typeErrorMsg.className = "inline-error-message";
    typeErrorMsg.style.color = "#dc3545";
    typeErrorMsg.style.fontSize = "12px";
    typeErrorMsg.style.display = "none";
    typeErrorMsg.dataset.columnName = col.name;
    typeErrorMsg.dataset.errorType = "type";

    typeWrapper.appendChild(select);
    typeWrapper.appendChild(typeErrorMsg);
    targetTypeTd.appendChild(typeWrapper);

    // Action (two buttons)
    const actionTd = document.createElement("td");
    const buttonWrapper = document.createElement("div");
    buttonWrapper.style.display = "flex";
    buttonWrapper.style.gap = "5px";
    buttonWrapper.style.flexDirection = "column";

    // Rename button
    const renameBtn = document.createElement("button");
    renameBtn.className = "btn btn-secondary";
    renameBtn.textContent = "Rename";
    renameBtn.disabled = true;
    renameBtn.dataset.columnName = col.name;
    renameBtn.dataset.action = "save-rename";
    renameBtn.style.padding = "4px 8px";
    renameBtn.style.fontSize = "11px";

    // Type change button
    const typeBtn = document.createElement("button");
    typeBtn.className = "btn btn-secondary";
    typeBtn.textContent = "Change Type";
    typeBtn.disabled = true;
    typeBtn.dataset.columnName = col.name;
    typeBtn.dataset.action = "save-type";
    typeBtn.style.padding = "4px 8px";
    typeBtn.style.fontSize = "11px";

    // Event listeners for enabling/disabling buttons
    nameInput.addEventListener("input", () => {
      const newName = nameInput.value.trim();
      const isChanged = newName && newName !== col.name;
      renameBtn.disabled = !isChanged;
      renameBtn.className = isChanged ? "btn btn-primary" : "btn btn-secondary";
      
      // Clear error message when user types
      nameErrorMsg.style.display = "none";
      nameErrorMsg.textContent = "";
    });

    select.addEventListener("change", () => {
      const isChanged = select.value !== col.current_type;
      typeBtn.disabled = !isChanged;
      typeBtn.className = isChanged ? "btn btn-primary" : "btn btn-secondary";
      
      // Clear error message when user changes selection
      typeErrorMsg.style.display = "none";
      typeErrorMsg.textContent = "";
    });

    buttonWrapper.appendChild(renameBtn);
    buttonWrapper.appendChild(typeBtn);
    actionTd.appendChild(buttonWrapper);

    tr.appendChild(nameTd);
    tr.appendChild(currentNameTd);
    tr.appendChild(newNameTd);
    tr.appendChild(typeTd);
    tr.appendChild(targetTypeTd);
    tr.appendChild(actionTd);
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

export function renderRenameControls(container, columns) {
  if (!container) return;
  container.innerHTML = "";
  const cols = columns || [];
  if (!cols.length) {
    const p = document.createElement("p");
    p.className = "placeholder-description";
    p.textContent = "Upload a dataset to rename columns.";
    container.appendChild(p);
    return;
  }

  cols.forEach((col) => {
    const row = document.createElement("div");
    row.className = "rename-row";

    const label = document.createElement("div");
    label.className = "rename-label";
    label.textContent = col.name;

    const input = document.createElement("input");
    input.type = "text";
    input.className = "field-input";
    input.placeholder = "New name";
    input.value = col.name;
    input.dataset.columnName = col.name;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn btn-secondary";
    button.textContent = "Rename";
    button.dataset.columnName = col.name;

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(button);

    container.appendChild(row);
  });
}

export function renderMissingValues(container, missingSummary) {
  if (!container) return;
  container.innerHTML = "";
  const rows = missingSummary || [];
  if (!rows.length) {
    const p = document.createElement("p");
    p.className = "placeholder-description";
    p.textContent = "No missing value information available. Upload a dataset first.";
    container.appendChild(p);
    return;
  }

  rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = "missing-row";

    const name = document.createElement("div");
    name.className = "missing-name";
    name.textContent = row.column_name;

    const stats = document.createElement("div");
    stats.className = "missing-stats";
    stats.textContent = `${row.missing_count} missing (${row.missing_ratio ?? "-"}%)`;

    const select = document.createElement("select");
    select.className = "field-input field-input-inline";
    select.dataset.columnName = row.column_name;
    [
      "Leave as-is",
      "Drop rows",
      "Drop column",
      "Fill with mean",
      "Fill with median",
      "Fill with mode",
    ].forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      select.appendChild(option);
    });

    line.appendChild(name);
    line.appendChild(stats);
    line.appendChild(select);
    container.appendChild(line);
  });
}

export function renderLogs(container, logs) {
  if (!container) return;
  container.innerHTML = "";
  const entries = logs || [];
  if (!entries.length) {
    const p = document.createElement("p");
    p.className = "placeholder-description";
    p.textContent = "No recent log entries.";
    container.appendChild(p);
    return;
  }

  entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "log-row";

    const icon = document.createElement("div");
    icon.className = "log-icon";

    const text = document.createElement("div");
    text.className = "log-text";

    const title = document.createElement("div");
    title.className = "log-title";
    title.textContent = entry.message || "Log entry";

    const meta = document.createElement("div");
    meta.className = "log-meta";
    const ts = entry.timestamp || "";
    const level = (entry.level || "info").toUpperCase();
    meta.textContent = `${ts} • ${level}`;

    text.appendChild(title);
    text.appendChild(meta);

    row.appendChild(icon);
    row.appendChild(text);
    container.appendChild(row);
  });
}
export function renderChangesPreview(container, previewData) {
  if (!container) return;
  
  const { data = [], columns = [] } = previewData || {};
  container.innerHTML = "";

  if (!columns.length) {
    const empty = document.createElement("p");
    empty.className = "placeholder-description";
    empty.textContent = "No changes made yet. Make a column change to see the preview.";
    container.appendChild(empty);
    return;
  }

  // Add title
  const title = document.createElement("h3");
  title.textContent = `Changes Preview (Random 30 Rows)`;
  title.style.marginBottom = "10px";
  container.appendChild(title);

  // Add metadata
  const meta = document.createElement("p");
  meta.className = "card-subtitle";
  meta.textContent = `Showing ${data.length} of ${previewData.total_rows} total rows from modified dataset`;
  meta.style.marginBottom = "15px";
  container.appendChild(meta);

  const table = document.createElement("table");
  table.className = "ds-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  data.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = row[col];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}