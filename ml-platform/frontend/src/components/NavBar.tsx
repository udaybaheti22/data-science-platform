import type { TabId } from "../types/api";
import "../styles/navbar.css";

interface NavBarProps {
  activeTab: TabId;
  isLoaded: boolean;
  onTabChange: (tab: TabId) => void;
}

const TABS: { id: TabId; label: string }[] = [
  { id: "upload", label: "Upload" },
  { id: "clean", label: "Clean" },
  { id: "analyze", label: "Analyze" },
  { id: "model", label: "Model" },
  { id: "suggest", label: "AI Suggestions" },
  { id: "export", label: "Export" },
];

export default function NavBar({ activeTab, isLoaded, onTabChange }: NavBarProps) {
  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      {TABS.map(({ id, label }) => {
        const enabled = id === "upload" || isLoaded;
        return (
          <button
            key={id}
            className={`nav-tab${activeTab === id ? " active" : ""}${!enabled ? " disabled" : ""}`}
            aria-disabled={!enabled}
            onClick={() => {
              if (enabled) onTabChange(id);
            }}
          >
            {label}
          </button>
        );
      })}
    </nav>
  );
}
