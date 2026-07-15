interface PreviewTableProps {
  columns: string[];
  data: Record<string, unknown>[];
}

export default function PreviewTable({ columns, data }: PreviewTableProps) {
  if (!data.length) return <p className="empty-state">No data to display.</p>;

  return (
    <div className="preview-table-wrapper">
      <table className="preview-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col}>
                  {row[col] === null || row[col] === undefined
                    ? <span className="null-value">NaN</span>
                    : String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
