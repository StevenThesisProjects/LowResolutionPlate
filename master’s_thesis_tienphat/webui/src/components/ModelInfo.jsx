import { pct } from '../format.js';

export default function ModelInfo({ model }) {
  if (!model) return null;
  const frames = ['1', '2', '3', '4', '5'];
  return (
    <details className="card model-card">
      <summary>Thông tin model và ngưỡng phân loại</summary>
      <dl className="kv">
        <dt>Phiên bản</dt><dd><code>{model.model_version}</code> ({model.parameters.toLocaleString('vi-VN')} tham số)</dd>
        <dt>SHA-256</dt><dd>{model.sha256_verified ? 'đã kiểm, khớp checkpoint đóng băng' : 'KHÔNG khớp'}</dd>
        <dt>Bộ giải mã mặc định</dt><dd>{model.default_decoder}</dd>
        <dt>Cách chọn checkpoint</dt><dd>{model.selection_rule}</dd>
        <dt>Phạm vi</dt><dd>{model.scope}</dd>
      </dl>
      <div className="table-wrap">
        <table className="table table--compact">
          <thead><tr><th>Số frame</th>{frames.map((n) => <th key={n}>{n}</th>)}</tr></thead>
          <tbody>
            <tr><td>Ngưỡng “tự động”</td>{frames.map((n) => <td key={n}>{pct(model.thresholds_by_frames[n], 2)}</td>)}</tr>
            <tr><td>Accuracy lock-box</td>{frames.map((n) => <td key={n}>{model.expected_accuracy_by_frames[n]?.toFixed(1)}%</td>)}</tr>
          </tbody>
        </table>
      </div>
    </details>
  );
}
