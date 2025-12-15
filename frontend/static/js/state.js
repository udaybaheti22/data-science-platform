window.appState = {
  currentSection: "upload",
  datasetMeta: null,
  checkpoints: [],
  loading: {}
};

function setCurrentSection(section) {
  window.appState.currentSection = section;
}

function setDatasetMeta(meta) {
  window.appState.datasetMeta = meta;
}

function setCheckpoints(list) {
  window.appState.checkpoints = list;
}

function setLoadingFlag(key, value) {
  window.appState.loading[key] = !!value;
}

window.state = { setCurrentSection, setDatasetMeta, setCheckpoints, setLoadingFlag };
