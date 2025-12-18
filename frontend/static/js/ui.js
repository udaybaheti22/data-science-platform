// UI helpers for section switching and navigation

export function showSection(sectionId) {
  const sections = document.querySelectorAll(".section");
  sections.forEach((sec) => {
    sec.style.display = sec.id === sectionId ? "block" : "none";
  });
}

export function setActiveSidebar(targetSectionId) {
  const items = document.querySelectorAll(".sidebar-item");
  items.forEach((item) => {
    const section = item.getAttribute("data-section");
    if (section === targetSectionId) {
      item.classList.add("sidebar-item-active");
    } else {
      item.classList.remove("sidebar-item-active");
    }
  });
}

export function showInfo(message) {
  const box = document.getElementById("alert-info");
  if (!box) return;
  box.textContent = message || "";
  box.style.display = message ? "block" : "none";
}

export function showError(message) {
  const box = document.getElementById("alert-error");
  if (!box) return;
  box.textContent = message || "";
  box.style.display = message ? "block" : "none";
}

export function clearAlerts() {
  showInfo("");
  showError("");
}
