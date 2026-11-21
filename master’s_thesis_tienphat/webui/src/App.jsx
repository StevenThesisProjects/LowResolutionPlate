import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, getApiKey, setApiKey } from './api.js';
import { framePreviews } from './frames.js';
import InputPanel from './components/InputPanel.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import History from './components/History.jsx';
import ModelInfo from './components/ModelInfo.jsx';

// crypto.randomUUID needs a secure context; getRandomValues also works over plain http (LAN demo)
const newId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)),
  (b) => b.toString(16).padStart(2, '0')).join('');

function sessionId() {
  try {
    let id = sessionStorage.getItem('mfplate.session');
    if (!id) { id = newId(); sessionStorage.setItem('mfplate.session', id); }
    return id;
  } catch {
    return newId();
  }
}

// Object URLs of previews stay alive while either the current selection or the shown
// result uses them.
function release(list, keep = []) {
  const kept = new Set(keep.map((p) => p.url));
  list.forEach((p) => { if (!kept.has(p.url)) URL.revokeObjectURL(p.url); });
}

export default function App() {
  const session = useMemo(sessionId, []);
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);
  const [samples, setSamples] = useState([]);
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [decoder, setDecoder] = useState('default');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [needKey, setNeedKey] = useState(false);
  const [keyDraft, setKeyDraft] = useState(getApiKey());
  const resultRef = useRef(null);
  const previewsRef = useRef([]);
  resultRef.current = result;
  previewsRef.current = previews;

  const fail = useCallback((e) => {
    if (e.status === 401) setNeedKey(true);
    setError(e);
  }, []);

  const refreshHistory = useCallback(() => {
    api.history(session).then(setHistory).catch(() => {});
  }, [session]);

  const loadMeta = useCallback(async () => {
    setHealth(await api.health());
    try { setModel(await api.model()); } catch (e) { if (e.status === 401) setNeedKey(true); }
    setSamples(await api.samples());
    refreshHistory();
  }, [refreshHistory]);

  const selectFiles = useCallback(async (list) => {
    const made = await framePreviews(list).catch(() => []);
    setFiles(list);
    setError(null);
    release(previewsRef.current, resultRef.current?.previews);
    setPreviews(made);
    return made;
  }, []);

  const run = useCallback(async (inputFiles, inputPreviews) => {
    if (!inputFiles.length) return;
    setBusy(true);
    setError(null);
    try {
      const data = await api.predict(inputFiles, { decoder, sessionId: session });
      const old = resultRef.current;
      if (old) release(old.previews, inputPreviews.concat(previewsRef.current));
      setResult({ data, previews: inputPreviews });
      refreshHistory();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }, [decoder, session, refreshHistory, fail]);

  const loadSample = useCallback(async (name, autoRun = true) => {
    try {
      const file = await api.sampleFile(name);
      const made = await selectFiles([file]);
      if (autoRun) await run([file], made);
    } catch (e) {
      fail(e);
    }
  }, [selectFiles, run, fail]);

  useEffect(() => {
    loadMeta();
  }, [loadMeta]);

  // ?sample=<name>.zip opens the page with that demo track already read
  const autoSample = useRef(false);
  useEffect(() => {
    const name = new URLSearchParams(window.location.search).get('sample');
    if (name && samples.some((s) => s.name === name) && !autoSample.current) {
      autoSample.current = true;
      loadSample(name, true);
    }
  }, [samples, loadSample]);

  const ready = health?.status === 'ok';
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand__logo" aria-hidden="true" />
          <div>
            <h1>MF-PlateReader</h1>
            <p>Đọc biển số độ phân giải thấp từ nhiều khung hình</p>
          </div>
        </div>
        <div className={`health ${ready ? 'is-ok' : 'is-down'}`} title={health?.detail || ''}>
          <span className="health__dot" aria-hidden="true" />
          {health == null ? 'Đang kết nối…' : ready ? 'Model sẵn sàng' : 'Model chưa sẵn sàng'}
          {model && <code className="health__ver">{model.model_version}</code>}
        </div>
      </header>

      <p className="scope">
        Chỉ hỗ trợ biển Brazil/Mercosur một dòng, 7 ký tự (bộ dữ liệu LRLPR-26), đầu vào là ảnh đã crop
        biển số. Không dùng cho biển Việt Nam. Kết quả chỉ để tham khảo.
      </p>

      {needKey && (
        <form className="card keyform" onSubmit={(e) => {
          e.preventDefault(); setApiKey(keyDraft.trim()); setNeedKey(false); setError(null); loadMeta();
        }}>
          <label htmlFor="apikey">API đang bật khoá truy cập. Nhập khoá:</label>
          <input id="apikey" type="password" value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)} />
          <button type="submit" className="btn btn--primary">Lưu</button>
        </form>
      )}

      <main className="layout">
        <InputPanel
          files={files}
          previews={previews}
          onSelect={selectFiles}
          onRemove={(name) => selectFiles(files.filter((f) => f.name !== name))}
          onClear={() => selectFiles([])}
          samples={samples}
          onSample={(name) => loadSample(name, true)}
          decoder={decoder}
          onDecoder={setDecoder}
          onRun={() => run(files, previews)}
          busy={busy}
          maxFrames={model?.limits?.max_frames ?? 5}
        />
        <ResultPanel result={result} busy={busy} error={error} onFeedbackSent={refreshHistory} />
      </main>

      <History items={history} />
      <ModelInfo model={model} />

      <footer className="footer">
        API: <a href="/docs">/docs</a> · Ảnh không được lưu trên server; nhật ký chỉ giữ mã băm SHA-256 của ảnh.
      </footer>
    </div>
  );
}
