import { useState } from "react";
import NavBar from "./components/NavBar";
import UploadTab from "./tabs/UploadTab";
import CleanTab from "./tabs/CleanTab";
import AnalyzeTab from "./tabs/AnalyzeTab";
import ModelTab from "./tabs/ModelTab";
import ExportTab from "./tabs/ExportTab";
import type { TabId } from "./types/api";
import "./styles/global.css";

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("upload");
  const [isLoaded, setIsLoaded] = useState(false);
  const [filename, setFilename] = useState<string | null>(null);
  const [rowCount, setRowCount] = useState<number | null>(null);
  const [colCount, setColCount] = useState<number | null>(null);
  const [columns, setColumns] = useState<string[]>([]);

  const isTabEnabled = (tab: TabId): boolean => {
    if (tab === "upload") return true;
    return isLoaded;
  };

  function handleUploadSuccess(
    newFilename: string,
    newRows: number,
    newCols: number,
    newColumns: string[]
  ) {
    setFilename(newFilename);
    setRowCount(newRows);
    setColCount(newCols);
    setColumns(newColumns);
    setIsLoaded(true);
  }

  function handleColumnsChange(newColumns: string[], newRowCount?: number) {
    setColumns(newColumns);
    if (newRowCount !== undefined) setRowCount(newRowCount);
  }

  function handleClearDataset() {
    setIsLoaded(false);
    setFilename(null);
    setRowCount(null);
    setColCount(null);
    setColumns([]);
    setActiveTab("upload");
  }

  function renderTab() {
    switch (activeTab) {
      case "upload":
        return (
          <UploadTab
            isLoaded={isLoaded}
            filename={filename}
            rowCount={rowCount}
            colCount={colCount}
            onUploadSuccess={handleUploadSuccess}
            onClearDataset={handleClearDataset}
            onColumnsChange={handleColumnsChange}
          />
        );
      case "clean":
        return (
          <CleanTab
            columns={columns}
            rowCount={rowCount}
            onColumnsChange={handleColumnsChange}
          />
        );
      case "analyze":
        return <AnalyzeTab />;
      case "model":
        return <ModelTab columns={columns} />;
      case "export":
        return <ExportTab />;
      default:
        return null;
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>ML Platform</h1>
        {isLoaded && filename && (
          <span className="header-file-info">
            📄 {filename} — {rowCount} rows × {colCount} cols
          </span>
        )}
      </header>

      <NavBar
        activeTab={activeTab}
        isLoaded={isLoaded}
        onTabChange={(tab) => {
          if (isTabEnabled(tab)) setActiveTab(tab);
        }}
      />

      <main className="tab-content">{renderTab()}</main>
    </div>
  );
}

export default App;
