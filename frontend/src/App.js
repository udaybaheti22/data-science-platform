import React, { useEffect, useState } from 'https://esm.sh/react@18';
import ReactDOM from 'https://esm.sh/react-dom@18/client';
import { BuildModel, Button } from './components.js';

const App = () => {
  const [theme, setTheme] = useState('light');
  useEffect(() => { document.documentElement.classList.toggle('dark', theme === 'dark'); }, [theme]);
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-800/80 backdrop-blur border-b">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="text-lg font-bold text-gray-900 dark:text-white">Data Science Platform</div>
          <Button variant="secondary" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>Toggle {theme === 'dark' ? 'Light' : 'Dark'}</Button>
        </div>
      </header>
      <main className="max-w-7xl mx-auto p-4 space-y-6">
        <BuildModel />
      </main>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

export default App;


