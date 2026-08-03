import { useState } from "react";
import NavBar from "./components/NavBar";
import UploadTab from "./tabs/UploadTab";
import CleanTab from "./tabs/CleanTab";
import AnalyzeTab from "./tabs/AnalyzeTab";
import ModelTab from "./tabs/ModelTab";
import ExportTab from "./tabs/ExportTab";
import SuggestTab from "./tabs/SuggestTab";
import type { TabId, AISuggestionResponse } from "./types/api";
import "./styles/global.css";

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("upload");
  const [isLoaded, setIsLoaded] = useState(false);
  const [filename, setFilename] = useState<string | null>(null);
  const [rowCount, setRowCount] = useState<number | null>(null);
  const [colCount, setColCount] = useState<number | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestionResponse | null>(null);
  const [corrImg, setCorrImg] = useState<string | null>(null);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

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
    // Reset analyze cache — new dataset means old results are stale
    setCorrImg(null);
    setReportUrl(null);
  }

  function handleColumnsChange(newColumns: string[], newRowCount?: number) {
    setColumns(newColumns);
    if (newRowCount !== undefined) setRowCount(newRowCount);
    // Reset analyze cache — cleaning changed the dataset
    setCorrImg(null);
    setReportUrl(null);
  }

  function handleClearDataset() {
    setIsLoaded(false);
    setFilename(null);
    setRowCount(null);
    setColCount(null);
    setColumns([]);
    setAiSuggestions(null);
    setCorrImg(null);
    setReportUrl(null);
    setActiveTab("upload");
  }

  function handleSuggestionsGenerated(suggestions: AISuggestionResponse) {
    setAiSuggestions(suggestions);
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
            aiSuggestions={aiSuggestions}
          />
        );
      case "analyze":
        return (
          <AnalyzeTab
            corrImg={corrImg}
            reportUrl={reportUrl}
            onCorrImgChange={setCorrImg}
            onReportUrlChange={setReportUrl}
          />
        );
      case "model":
        return <ModelTab columns={columns} aiSuggestions={aiSuggestions} />;
      case "suggest":
        return (
          <SuggestTab
            columns={columns}
            aiSuggestions={aiSuggestions}
            onSuggestionsGenerated={handleSuggestionsGenerated}
          />
        );
      case "export":
        return <ExportTab />;
      default:
        return null;
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Data Science Platform</h1>
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

      <footer className="footer">
        <span className="footer-credit">Project By — <strong>Uday Baheti</strong></span>
        <span className="footer-divider">|</span>
        <a href="https://github.com/udaybaheti22" target="_blank" rel="noopener noreferrer">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.089-.744.084-.729.084-.729 1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.76-1.605-2.665-.3-5.467-1.332-5.467-5.93 0-1.31.468-2.38 1.235-3.22-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.3 1.23a11.52 11.52 0 0 1 3.003-.404c1.02.005 2.045.138 3.003.404 2.29-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.234 1.91 1.234 3.22 0 4.61-2.807 5.625-5.48 5.92.43.372.823 1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 0 .319.216.694.825.576C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
          </svg>
          GitHub
        </a>
        <span className="footer-divider">|</span>
        <a href="https://www.linkedin.com/in/udaybaheti" target="_blank" rel="noopener noreferrer">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
          </svg>
          LinkedIn
        </a>
      </footer>
    </div>
  );
}

export default App;
