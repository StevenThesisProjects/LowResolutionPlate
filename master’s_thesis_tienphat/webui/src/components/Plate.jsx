import { confLevel, pct } from '../format.js';

// The reading drawn as a plate, with each character's confidence underneath.
// When the zip carried a label, mismatching characters are marked.
export default function Plate({ text, layout, charConf, reference }) {
  const chars = (text || '').split('');
  const cols = Math.max(chars.length, 1);
  const mismatch = (i) => reference && reference[i] !== chars[i];
  return (
    <div className="plate-block">
      <div className={`plate plate--${(layout || 'unknown').toLowerCase()}`}>
        <div className="plate__band">{layout || 'Không đúng mẫu biển'}</div>
        <div className="plate__chars" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {chars.length ? chars.map((c, i) => (
            <span key={i} className={`plate__char${mismatch(i) ? ' is-wrong' : ''}`}>{c}</span>
          )) : <span className="plate__char muted">—</span>}
        </div>
      </div>
      <div className="charconf" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        aria-label="Độ tin cậy từng ký tự">
        {chars.map((c, i) => (
          <div key={i} className={`charconf__cell lvl-${confLevel(charConf[i] ?? 0)}`} title={`${c}: ${pct(charConf[i], 2)}`}>
            <div className="charconf__bar"><i style={{ height: `${Math.max(4, (charConf[i] ?? 0) * 100)}%` }} /></div>
            <span>{pct(charConf[i], 0)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
