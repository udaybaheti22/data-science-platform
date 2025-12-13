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

window.render = { renderPreview };
