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
  ["Column", "Current type", "Non-null", "New type"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  const tbody = document.createElement("tbody");
  cols.forEach((col) => {
    const tr = document.createElement("tr");

    const nameTd = document.createElement("td");
    nameTd.textContent = col.name;

    const typeTd = document.createElement("td");
    typeTd.textContent = col.current_type;

    const nonNullTd = document.createElement("td");
    nonNullTd.textContent = col.non_null_count;

    const newTypeTd = document.createElement("td");
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
    newTypeTd.appendChild(select);

    tr.appendChild(nameTd);
    tr.appendChild(typeTd);
    tr.appendChild(nonNullTd);
    tr.appendChild(newTypeTd);
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
