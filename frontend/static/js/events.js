function wireEvents(root) {
  const fileInput = root.querySelector("#fileInput");
  const uploadBtn = root.querySelector("#uploadBtn");
  const previewContainer = root.querySelector("#preview");

  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    uploadBtn.disabled = true;
    try {
      await window.api.uploadFile(file);
      const preview = await window.api.getPreview();
      window.render.renderPreview(previewContainer, preview);
    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      uploadBtn.disabled = false;
    }
  });
}

window.events = { wireEvents };
