import { useState } from 'react';
import { api } from '../api.js';
import { pct, REASONS } from '../format.js';
import Plate from './Plate.jsx';

function Status({ data }) {
  const review = data.needs_review;
  const t = data.triage;
  return (
    <div className={`status ${review ? 'status--review' : 'status--auto'}`}>
      <div className="status__head">
        <span className="status__dot" aria-hidden="true" />
        <strong>{review ? 'Cần người kiểm tra' : 'Có thể dùng tự động'}</strong>
      </div>
      {review ? (
        <ul className="status__reasons">
          {data.review_reasons.map((r) => (
            <li key={r}>{(REASONS[r] || (() => r))(data.n_frames)}</li>
          ))}
        </ul>
      ) : (
        <p>Độ tin cậy vượt ngưỡng đã hiệu chỉnh cho {data.n_frames} frame.</p>
      )}
      {t.threshold != null && (
        <div className="gauge" title={`điểm ${t.score.toFixed(4)} · ngưỡng ${t.threshold.toFixed(4)}`}>
          <div className="gauge__track">
            <i className="gauge__fill" style={{ width: `${t.score * 100}%` }} />
            <i className="gauge__mark" style={{ left: `${t.threshold * 100}%` }} />
          </div>
          <span className="small muted">Điểm {pct(t.score, 2)} · ngưỡng {pct(t.threshold, 2)}</span>
        </div>
      )}
    </div>
  );
}

function Reference({ plate, reference }) {
  if (!reference) return null;
  const ok = plate === reference;
  return (
    <p className={`reference ${ok ? 'is-ok' : 'is-wrong'}`}>
      Nhãn đúng (từ zip): <code>{reference.split('').map((c, i) => (
        <span key={i} className={plate[i] !== c ? 'diff' : ''}>{c}</span>
      ))}</code> {ok ? '— khớp' : '— model đọc sai'}
    </p>
  );
}

function TopK({ candidates }) {
  const best = candidates[0]?.plate || '';
  return (
    <ol className="topk">
      {candidates.map((c, i) => (
        <li key={c.plate} className={i === 0 ? 'is-best' : ''}>
          <code className="topk__plate">
            {c.plate.split('').map((ch, j) => (
              <span key={j} className={i > 0 && best[j] !== ch ? 'diff' : ''}>{ch}</span>
            ))}
          </code>
          <span className="topk__bar"><i style={{ width: `${Math.max(1, c.prob * 100)}%` }} /></span>
          <span className="topk__prob">{pct(c.prob, c.prob < 0.01 ? 2 : 1)}</span>
        </li>
      ))}
    </ol>
  );
}

function FramesUsed({ frames, previews }) {
  const max = Math.max(...frames.map((f) => f.weight));
  const urlOf = (f) => (previews.find((p) => p.name === f.name) || previews[f.index] || {}).url;
  return (
    <ul className="frames">
      {frames.map((f) => (
        <li key={f.index} className={`frame${f.weight === max && frames.length > 1 ? ' is-top' : ''}`}>
          <div className="frame__img">{urlOf(f) ? <img src={urlOf(f)} alt={f.name} /> : <span>{f.name}</span>}</div>
          <div className="frame__meta">
            <span className="frame__name" title={f.name}>{f.name}</span>
            <span className="frame__bar"><i style={{ width: `${f.weight * 100}%` }} /></span>
            <span className="small">Trọng số <strong>{pct(f.weight, 0)}</strong></span>
            <span className="small muted">Độ nét {Math.round(f.sharpness)} · {f.width}×{f.height}px</span>
          </div>
        </li>
      ))}
    </ul>
  );
}

function Feedback({ data, onSent }) {
  const [mode, setMode] = useState('ask');            // ask | correcting | sent | error
  const [value, setValue] = useState(data.reference_plate || data.plate);
  const [message, setMessage] = useState('');
  const send = async (correct) => {
    try {
      await api.feedback(data.request_id, correct, correct ? null : value.trim().toUpperCase());
      setMode('sent');
      setMessage(correct ? 'Đã ghi nhận: kết quả đúng.' : `Đã ghi nhận biển số đúng là ${value.trim().toUpperCase()}.`);
      onSent();
    } catch (e) {
      setMode('error');
      setMessage(e.message);
    }
  };
  if (mode === 'sent') return <p className="feedback__done">✓ {message}</p>;
  return (
    <div className="feedback">
      <span>Kết quả này đúng không?</span>
      {mode !== 'correcting' ? (
        <div className="feedback__actions">
          <button type="button" className="btn btn--ok" onClick={() => send(true)}>Đúng</button>
          <button type="button" className="btn btn--bad" onClick={() => setMode('correcting')}>Sai</button>
        </div>
      ) : (
        <form className="feedback__form" onSubmit={(e) => { e.preventDefault(); send(false); }}>
          <input value={value} onChange={(e) => setValue(e.target.value)} maxLength={10}
            pattern="[A-Za-z0-9]{1,10}" aria-label="Biển số đúng" autoFocus />
          <button type="submit" className="btn btn--primary">Gửi</button>
          <button type="button" className="link" onClick={() => setMode('ask')}>Huỷ</button>
        </form>
      )}
      {mode === 'error' && <p className="alert alert--error small">{message}</p>}
    </div>
  );
}

export default function ResultPanel({ result, busy, error, onFeedbackSent }) {
  const data = result?.data;
  return (
    <section className="card result-card" aria-labelledby="result-title" aria-live="polite">
      <div className="card__head">
        <h2 id="result-title"><span className="step">2</span> Kết quả</h2>
        {data && <span className="muted small">{data.n_frames} frame · {data.decoder} · {Math.round(data.latency_ms)} ms</span>}
      </div>

      {error && (
        <div className="alert alert--error">
          <strong>Không đọc được</strong> <code>{error.code}</code>
          <p>{error.message}</p>
        </div>
      )}

      {!data && !busy && !error && (
        <div className="empty">
          <div className="empty__plate" aria-hidden="true"><span>ABC1D23</span></div>
          <p>Chọn ảnh hoặc một track mẫu rồi bấm <strong>Đọc biển số</strong>.</p>
          <p className="small muted">Kết quả gồm biển số, độ tin cậy từng ký tự, các phương án khác và
            frame nào đóng góp nhiều nhất.</p>
        </div>
      )}

      {busy && !data && <div className="skeleton" aria-label="Đang đọc" />}

      {data && (
        <div className={`result${busy ? ' is-stale' : ''}`}>
          <div className="result__top">
            <Plate text={data.plate} layout={data.layout} charConf={data.char_conf} reference={data.reference_plate} />
            <Status data={data} />
          </div>
          <Reference plate={data.plate} reference={data.reference_plate} />

          <div className="stats">
            <div className="stat"><span className="stat__label">Độ tin cậy</span><span className="stat__value">{pct(data.confidence, 2)}</span><span className="stat__hint">trung bình theo ký tự</span></div>
            <div className="stat"><span className="stat__label">Xác suất cả chuỗi</span><span className="stat__value">{pct(data.sequence_prob, 1)}</span><span className="stat__hint">theo CTC</span></div>
            <div className="stat"><span className="stat__label">Accuracy kỳ vọng</span><span className="stat__value">{data.expected_accuracy != null ? `${data.expected_accuracy.toFixed(1)}%` : '—'}</span><span className="stat__hint">lock-box, {data.n_frames} frame</span></div>
            <div className="stat"><span className="stat__label">Thời gian</span><span className="stat__value">{Math.round(data.latency_ms)} ms</span><span className="stat__hint">model {Math.round(data.timing.model_ms)} ms</span></div>
          </div>

          {data.warnings.length > 0 && (
            <ul className="alert alert--warn warnings">
              {data.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}

          <div className="split">
            <div>
              <h3>Các phương án</h3>
              <TopK candidates={data.top3} />
              <p className="small muted">Ký tự tô màu là chỗ khác với kết quả chính.</p>
            </div>
            <div>
              <h3>Đóng góp của từng frame</h3>
              <FramesUsed frames={data.frames} previews={result.previews} />
            </div>
          </div>

          <Feedback key={data.request_id} data={data} onSent={onFeedbackSent} />

          <details className="details">
            <summary>Chi tiết kỹ thuật</summary>
            <dl>
              <dt>Mã yêu cầu</dt><dd><code>{data.request_id}</code></dd>
              <dt>Model</dt><dd><code>{data.model_version}</code></dd>
              <dt>Thời gian</dt><dd>đọc file {data.timing.read_ms} ms · model {data.timing.model_ms} ms · giải mã {data.timing.decode_ms} ms</dd>
            </dl>
            <pre>{JSON.stringify(data, null, 2)}</pre>
          </details>
        </div>
      )}
    </section>
  );
}
