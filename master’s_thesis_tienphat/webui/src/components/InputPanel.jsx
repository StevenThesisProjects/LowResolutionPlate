import { useRef, useState } from 'react';
import { isZip } from '../frames.js';
import { DECODERS, sampleLabel } from '../format.js';

export default function InputPanel({
  files, previews, onSelect, onRemove, onClear, samples, onSample, decoder, onDecoder, onRun, busy, maxFrames,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const zips = files.filter(isZip);
  const problem = zips.length && files.length > 1
    ? 'Gửi 1 file zip hoặc 1–5 ảnh, không gửi lẫn.'
    : !zips.length && files.length > maxFrames
      ? `Đang chọn ${files.length} ảnh; tối đa ${maxFrames} ảnh của cùng một xe.`
      : null;

  const pick = (list) => {
    const arr = Array.from(list || []);
    if (arr.length) onSelect(arr);
  };

  return (
    <section className="card input-card" aria-labelledby="input-title">
      <div className="card__head">
        <h2 id="input-title"><span className="step">1</span> Ảnh đầu vào</h2>
        {files.length > 0 && <button type="button" className="link" onClick={onClear}>Xoá hết</button>}
      </div>

      <div
        className={`dropzone${dragging ? ' is-drag' : ''}${files.length ? ' has-files' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.zip,image/jpeg,image/png,application/zip"
          hidden
          onChange={(e) => { pick(e.target.files); e.target.value = ''; }}
        />
        <svg className="dropzone__icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
        <p className="dropzone__title">Kéo thả hoặc bấm để chọn ảnh</p>
        <p className="muted small">1–5 ảnh crop biển số (JPG/PNG) của cùng một xe, hoặc 1 file zip của track</p>
      </div>

      {files.length > 0 && (
        <div className="selection">
          {zips.length === 1 && (
            <p className="small muted">
              <strong>{zips[0].name}</strong> · {previews.length} frame LR (ảnh HR trong zip được bỏ qua)
            </p>
          )}
          <ul className="thumbs">
            {previews.map((p) => (
              <li key={p.url} className="thumb">
                <img src={p.url} alt={p.name} />
                <span className="thumb__name" title={p.name}>{p.name}</span>
                {!zips.length && (
                  <button type="button" className="thumb__remove" aria-label={`Bỏ ${p.name}`}
                    onClick={() => onRemove(p.name)}>×</button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {samples.length > 0 && (
        <div className="field">
          <span className="label">Hoặc thử track mẫu</span>
          <div className="chips">
            {samples.map((s) => (
              <button key={s.name} type="button" className="chip" title={s.name}
                onClick={() => onSample(s.name)} disabled={busy}>
                {sampleLabel(s.name)}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="field">
        <span className="label" id="decoder-label">Bộ giải mã</span>
        <div className="segmented" role="radiogroup" aria-labelledby="decoder-label">
          {DECODERS.map(([value, label]) => (
            <button key={value} type="button" role="radio" aria-checked={decoder === value}
              className={decoder === value ? 'is-on' : ''} onClick={() => onDecoder(value)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {problem && <p className="alert alert--warn small">{problem}</p>}

      <button type="button" className="btn btn--primary btn--block" onClick={onRun}
        disabled={busy || !files.length || !!problem}>
        {busy ? <><span className="spinner" aria-hidden="true" /> Đang đọc…</> : 'Đọc biển số'}
      </button>
    </section>
  );
}
