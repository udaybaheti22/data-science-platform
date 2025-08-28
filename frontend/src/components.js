import React, { useState, useEffect } from 'https://esm.sh/react@18';
import { trainModel, getProfile, exportCsv } from './api.js';

export const Card = ({ children, className = "" }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden ${className}`}>
    <div className="p-6">{children}</div>
  </div>
);

export const Button = ({ children, onClick, variant = "primary", className = "", disabled = false }) => {
  const base = "font-semibold py-3 px-6 rounded-lg transition duration-300 disabled:opacity-50";
  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-200 text-gray-800 hover:bg-gray-300",
    outline: "border-2 border-blue-600 text-blue-600 hover:bg-blue-50",
    success: "bg-green-600 text-white hover:bg-green-700"
  };
  return (
    <button onClick={onClick} className={`${base} ${variants[variant]} ${className}`} disabled={disabled}>{children}</button>
  );
};

export const ExportPanel = () => {
  const handleExport = async (type = 'full') => {
    const blob = await exportCsv(type);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = type === 'train' ? 'train_dataset.csv' : type === 'test' ? 'test_dataset.csv' : 'dataset.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card>
        <div className="text-center p-6">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">📊</span></div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Full Dataset</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">Export the complete processed dataset</p>
          <Button onClick={() => handleExport('full')} variant="primary" className="w-full">Download Full Dataset</Button>
        </div>
      </Card>
      <Card>
        <div className="text-center p-6">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">🎯</span></div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Training Set</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">Export the training portion</p>
          <Button onClick={() => handleExport('train')} variant="success" className="w-full">Download Training Set</Button>
        </div>
      </Card>
      <Card>
        <div className="text-center p-6">
          <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4"><span className="text-2xl">🧪</span></div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Test Set</h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">Export the test portion</p>
          <Button onClick={() => handleExport('test')} variant="outline" className="w-full">Download Test Set</Button>
        </div>
      </Card>
    </div>
  );
};

export const BuildModel = () => {
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState([]);
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [fitIntercept, setFitIntercept] = useState(true);
  const [positive, setPositive] = useState(false);
  const [columns, setColumns] = useState([]);
  const [numericalColumns, setNumericalColumns] = useState([]);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(true);

  useEffect(() => { (async () => { const p = await getProfile(); setColumns(p.column_info.map(c => c.Column)); setNumericalColumns(p.numerical_columns || []); })().catch(()=>{}); }, []);

  const toggleFeature = (c) => setFeatureColumns(prev => prev.includes(c) ? prev.filter(x=>x!==c) : [...prev, c]);

  const submit = async () => {
    setTraining(true); setError(null); setResult(null);
    try {
      const payload = { target_column: targetColumn, feature_columns: featureColumns, test_size: testSize, random_state: randomState, model_name: 'Linear Regression', hyperparameters: { fit_intercept: fitIntercept, positive } };
      const r = await trainModel(payload);
      setResult(r);
    } catch (e) {
      setError(e.detail || 'Training failed');
    } finally { setTraining(false); }
  };

  if (!expanded) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Select Columns</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Target Column (numerical)</label>
              <select value={targetColumn} onChange={e=>setTargetColumn(e.target.value)} className="w-full px-3 py-2 border border-gray-300 rounded-md">
                <option value="">Select target</option>
                {numericalColumns.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Feature Columns (numerical)</label>
              <div className="max-h-48 overflow-y-auto border rounded-md p-2">
                {columns.map(c => {
                  const isNum = numericalColumns.includes(c);
                  const sel = featureColumns.includes(c);
                  return (
                    <label key={c} className={`flex items-center space-x-2 py-1 ${!isNum ? 'opacity-50' : ''}`}>
                      <input type="checkbox" disabled={!isNum} checked={sel} onChange={()=>toggleFeature(c)} />
                      <span className={`${isNum ? 'text-gray-700 dark:text-gray-200' : 'text-gray-400'}`}>{c}{!isNum && ' (not numerical)'}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Train-Test Split</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Test %: {Math.round(testSize*100)}%</label>
              <input type="range" min="0.1" max="0.5" step="0.05" value={testSize} onChange={e=>setTestSize(parseFloat(e.target.value))} className="w-full" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Random State</label>
              <input type="number" value={randomState} onChange={e=>setRandomState(parseInt(e.target.value))} className="w-full px-3 py-2 border rounded-md" />
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Hyperparameters</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-2"><input type="checkbox" checked={fitIntercept} onChange={e=>setFitIntercept(e.target.checked)} /><span className="text-sm text-gray-700 dark:text-gray-300">Fit Intercept</span></label>
            <label className="flex items-center space-x-2"><input type="checkbox" checked={positive} onChange={e=>setPositive(e.target.checked)} /><span className="text-sm text-gray-700 dark:text-gray-300">Positive Coefficients</span></label>
          </div>
        </Card>

        <Button onClick={submit} variant="primary" className="w-full" disabled={training || !targetColumn || featureColumns.length===0}>{training ? 'Training...' : 'Train Model'}</Button>
      </div>

      <div className="space-y-6">
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Model Performance</h3>
          {error && <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">{error}</div>}
          {result ? (
            <div className="space-y-4">
              <div className="text-3xl font-bold text-blue-600">R-squared: {result.r2_score.toFixed(4)}</div>
              {typeof result.plot_url === 'string' && result.plot_url.startsWith('data:image/png;base64,') ? (
                <img src={result.plot_url} alt="Regression plot" className="rounded-md border" />
              ) : (
                <div className="text-sm text-gray-700 dark:text-gray-300">{result.plot_url || 'No plot available.'}</div>
              )}
            </div>
          ) : (
            <div className="text-gray-600 dark:text-gray-300">Train a model to see metrics and plot.</div>
          )}
        </Card>
        <ExportPanel />
      </div>
    </div>
  );
};


