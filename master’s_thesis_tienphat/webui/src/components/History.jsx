import { pct, timeOf } from '../format.js';

export default function History({ items }) {
  return (
    <section className="card history-card" aria-labelledby="history-title">
      <div className="card__head">
        <h2 id="history-title">Lịch sử trong phiên</h2>
        <span className="muted small">{items.length} lần đọc</span>
      </div>
      {items.length === 0 ? (
        <p className="muted small">Chưa có lần đọc nào trong phiên này.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Giờ</th><th>Biển số</th><th>Phân loại</th><th>Tin cậy</th><th>Frame</th><th>Phản hồi</th></tr>
            </thead>
            <tbody>
              {items.map((h) => (
                <tr key={h.request_id}>
                  <td>{timeOf(h.created_at)}</td>
                  <td><code>{h.plate || '—'}</code></td>
                  <td><span className={`tag ${h.needs_review ? 'tag--review' : 'tag--auto'}`}>{h.needs_review ? 'Cần kiểm tra' : 'Tự động'}</span></td>
                  <td>{pct(h.confidence, 1)}</td>
                  <td>{h.n_frames}</td>
                  <td>{h.feedback_correct == null ? <span className="muted">—</span>
                    : h.feedback_correct ? 'Đúng' : <>Sai{h.corrected_plate && <> → <code>{h.corrected_plate}</code></>}</>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
