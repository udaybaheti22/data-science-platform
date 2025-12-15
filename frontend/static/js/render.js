function renderPreview(container, preview) {
  const { data = [], columns = [] } = preview || {};
  container.innerHTML = "";
  const table = document.createElement("table");
  table.className = "min-w-full text-sm";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  columns.forEach((c) => {
    const th = document.createElement("th");
    th.className = "border px-2 py-1";
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement("tbody");
  data.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => {
      const td = document.createElement("td");
      td.className = "border px-2 py-1";
      td.textContent = row[c];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

function renderMetadata(container, meta) {
  container.textContent = `Rows: ${meta.total_rows} | Columns: ${meta.total_columns}`;
}

function renderError(container, message) {
  container.textContent = message || "Unknown error";
  container.classList.remove("hidden");
}

function clearError(container) {
  container.textContent = "";
  container.classList.add("hidden");
}

function showSection(id) {
  const sections = document.querySelectorAll(".dashboard-section");
  sections.forEach((s) => {
    if (s.id === id) s.classList.remove("hidden"); else s.classList.add("hidden");
  });
}

function setActiveSidebar(targetId) {
  const items = document.querySelectorAll(".sidebar-item");
  items.forEach((i) => {
    if (i.dataset.target === targetId) i.classList.add("active"); else i.classList.remove("active");
  });
}

function renderCleanDatasetInfo(container, missingSummary, columns) {
  container.innerHTML = "";
  if (!columns || columns.length === 0) {
    const p = document.createElement("p");
    p.textContent = "Upload a dataset to enable cleaning";
    container.appendChild(p);
    return;
  }
  const table = document.createElement("table");
  table.className = "min-w-full text-sm";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  ["Column", "Type", "Missing"].forEach((c) => {
    const th = document.createElement("th");
    th.className = "border px-2 py-1";
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement("tbody");
  const missingMap = {};
  (missingSummary || []).forEach((m) => { missingMap[m.column_name] = m.missing_count; });
  let typeMap = {};
  const profileInfo = window.appState.profileColumnInfo;
  if (Array.isArray(profileInfo)) {
    profileInfo.forEach((row) => {
      if (row && row.Column) typeMap[row.Column] = row.Dtype;
    });
  }
  columns.forEach((c) => {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td");
    td1.className = "border px-2 py-1";
    td1.textContent = c;
    const tdType = document.createElement("td");
    tdType.className = "border px-2 py-1";
    tdType.textContent = typeMap[c] || "";
    const td2 = document.createElement("td");
    td2.className = "border px-2 py-1";
    td2.textContent = missingMap[c] != null ? missingMap[c] : 0;
    tr.appendChild(td1);
    tr.appendChild(tdType);
    tr.appendChild(td2);
    tbody.appendChild(tr);
  });
  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);
}

function populateSelect(selectEl, options) {
  selectEl.innerHTML = "";
  (options || []).forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    selectEl.appendChild(o);
  });
}

function populateMultiSelect(selectEl, options) {
  populateSelect(selectEl, options);
}

function renderInfo(container, message) {
  container.textContent = message;
  container.classList.remove("hidden");
  container.classList.add("text-green-700");
}

function clearInfo(container) {
  container.textContent = "";
  container.classList.add("hidden");
  container.classList.remove("text-green-700");
}

function renderCheckpointsSelect(selectEl, checkpoints) {
  selectEl.innerHTML = "";
  const optFinal = document.createElement("option");
  optFinal.value = "final";
  optFinal.textContent = "Final Dataset";
  selectEl.appendChild(optFinal);
  checkpoints.forEach((cp) => {
    const o = document.createElement("option");
    o.value = cp.id;
    o.textContent = `${cp.id} (${cp.rows}x${cp.columns})`;
    selectEl.appendChild(o);
  });
}

function setCleanActive(active) {
  const cards = document.querySelectorAll("#cleanSection .card");
  cards.forEach((c) => {
    if (active) c.classList.remove("inactive-card"); else c.classList.add("inactive-card");
  });
}

function setButtonLoading(btn, loadingText) {
  if (!btn) return;
  btn.disabled = true;
  if (typeof loadingText === "string") btn.textContent = loadingText;
}

function resetButton(btn, normalText) {
  if (!btn) return;
  btn.disabled = false;
  if (typeof normalText === "string") btn.textContent = normalText;
}

function renderStage(stage) {
  const content = document.querySelector("#content");
  if (!content) return;
  content.innerHTML = "";
  if (stage === "upload") {
    content.innerHTML = `
      <section id="uploadSection" class="dashboard-section">
        <div class="card">
          <h2>Upload</h2>
          <div class="flex gap-2 items-center">
            <input id="fileInput" type="file" accept=".csv,.xlsx,.xls" />
            <button id="uploadBtn" class="btn btn-primary">Upload & Preview</button>
          </div>
        </div>
        <div class="card">
          <h2>Preview</h2>
          <div id="metaBox" class="mb-2 text-sm"></div>
          <div id="preview" class="table-container"></div>
        </div>
      </section>
    `;
    return;
  }
  if (stage === "clean") {
    content.innerHTML = `
      <section id="cleanSection" class="dashboard-section clean-section">
        <div class="card">
          <h2>Missing Values</h2>
          <div id="cleanDatasetInfo" class="mb-3"></div>
          <div class="space-y-2">
            <button id="dropRowsMissingBtn" class="btn">Drop rows with missing</button>
            <button id="dropColsMissingBtn" class="btn">Drop columns with missing</button>
            <div class="flex gap-2 items-center">
              <select id="fillMethod">
                <option value="mean">Mean</option>
                <option value="median">Median</option>
                <option value="mode">Mode</option>
                <option value="drop">Drop</option>
                <option value="constant">Constant</option>
              </select>
              <select id="fillColumns" multiple size="4" class="flex-1"></select>
              <input id="fillConstant" type="text" placeholder="Constant value" />
              <button id="fillMissingBtn" class="btn btn-primary">Apply fill</button>
            </div>
          </div>
        </div>
        <div class="card">
          <h2>Duplicates</h2>
          <button id="removeDuplicatesBtn" class="btn">Remove duplicates</button>
        </div>
        <div class="card">
          <h2>Column Operations</h2>
          <div class="flex gap-2 items-center mb-2">
            <select id="dropColumnsSelect" multiple size="4" class="flex-1"></select>
            <button id="dropColumnsBtn" class="btn">Drop Columns</button>
          </div>
          <div class="flex gap-2 items-center">
            <select id="renameSelect"></select>
            <input id="renameNew" type="text" placeholder="New name" />
            <button id="renameBtn" class="btn">Rename</button>
          </div>
        </div>
        <div class="card">
          <h2>Data Type Conversion</h2>
          <div class="flex gap-2 items-center">
            <select id="dtypeColumn"></select>
            <select id="dtypeNew">
              <option value="int64">int64</option>
              <option value="float64">float64</option>
              <option value="object">object</option>
              <option value="datetime64[ns]">datetime64[ns]</option>
            </select>
            <button id="changeTypeBtn" class="btn">Convert</button>
          </div>
        </div>
        <div class="card">
          <h2>Encoding</h2>
          <div class="flex gap-2 items-center mb-2">
            <select id="labelSelect" multiple size="4" class="flex-1"></select>
            <button id="labelEncodeBtn" class="btn">Label Encode</button>
          </div>
          <div class="flex gap-2 items-center">
            <select id="onehotSelect" multiple size="4" class="flex-1"></select>
            <button id="onehotEncodeBtn" class="btn">One-Hot Encode</button>
          </div>
        </div>
        <div class="card">
          <h2>Scaling</h2>
          <div class="flex gap-2 items-center">
            <select id="scaleSelect" multiple size="4" class="flex-1"></select>
            <select id="scaleMethod">
              <option value="standard">StandardScaler</option>
              <option value="minmax">MinMaxScaler</option>
            </select>
            <button id="scaleBtn" class="btn">Scale</button>
          </div>
        </div>
        <div class="card">
          <h2>Outliers</h2>
          <p>Not implemented yet.</p>
        </div>
        <div class="card">
          <h2>Save Dataset (Checkpoint)</h2>
          <div class="flex gap-2 items-center">
            <input id="checkpointDesc" type="text" placeholder="Description (optional)" class="flex-1" />
            <button id="saveCheckpointBtn" class="btn btn-primary">Save Dataset</button>
          </div>
        </div>
      </section>
    `;
    return;
  }
  if (stage === "analyze") {
    content.innerHTML = `
      <section id="analyzeSection" class="dashboard-section">
        <div class="card">
          <h2>Analyze</h2>
          <button id="profileBtn" class="btn btn-primary">Generate Data Profile</button>
        </div>
      </section>
    `;
    return;
  }
  if (stage === "model") {
    content.innerHTML = `
      <section id="modelSection" class="dashboard-section">
        <div class="card">
          <h2>Model</h2>
          <p>Modeling will be enabled in later iterations.</p>
        </div>
      </section>
    `;
    return;
  }
  if (stage === "export") {
    content.innerHTML = `
      <section id="exportSection" class="dashboard-section">
        <div class="card">
          <h2>Export</h2>
          <div class="flex gap-2 items-center">
            <select id="exportSource">
              <option value="final">Final Dataset</option>
            </select>
            <button id="exportBtn" class="btn btn-primary">Download CSV</button>
          </div>
        </div>
      </section>
    `;
    return;
  }
}

window.render = { renderPreview, renderMetadata, renderError, clearError, showSection, setActiveSidebar, renderCheckpointsSelect, renderCleanDatasetInfo, populateSelect, populateMultiSelect, renderInfo, clearInfo, setCleanActive, setButtonLoading, resetButton, renderStage };
