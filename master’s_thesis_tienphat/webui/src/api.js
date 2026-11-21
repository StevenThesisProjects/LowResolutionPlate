// Thin client for the MF-PlateReader REST API (same origin; see docs/api.md).
const KEY_STORE = 'mfplate.apiKey';

export function getApiKey() {
  try { return localStorage.getItem(KEY_STORE) || ''; } catch { return ''; }
}
export function setApiKey(value) {
  try { value ? localStorage.setItem(KEY_STORE, value) : localStorage.removeItem(KEY_STORE); } catch { /* private mode */ }
}

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function call(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = getApiKey();
  if (key) headers.set('X-API-Key', key);
  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch {
    throw new ApiError(0, 'network', 'Không kết nối được tới API.');
  }
  const isJson = (response.headers.get('content-type') || '').includes('application/json');
  if (!response.ok) {
    const body = isJson ? await response.json() : null;
    const err = (body && body.error) || {};
    throw new ApiError(response.status, err.code || 'http_error', err.message || `HTTP ${response.status}`);
  }
  return isJson ? response.json() : response;
}

export const api = {
  // /health answers 503 with a normal body while the model is not loaded
  health: () => fetch('/health').then((r) => r.json()).catch(() => ({ status: 'unavailable', model_loaded: false })),
  model: () => call('/model'),
  samples: () => call('/samples').catch(() => []),
  async sampleFile(name) {
    const response = await call(`/samples/${encodeURIComponent(name)}`);
    const blob = await response.blob();
    return new File([blob], name, { type: 'application/zip' });
  },
  predict(files, { decoder, sessionId, topK = 3 }) {
    const form = new FormData();
    files.forEach((f) => form.append('files', f, f.name));
    const query = new URLSearchParams({ top_k: String(topK) });
    if (decoder && decoder !== 'default') query.set('decoder', decoder);
    if (sessionId) query.set('session_id', sessionId);
    return call(`/predict?${query}`, { method: 'POST', body: form });
  },
  feedback(requestId, correct, correctedPlate) {
    const body = { request_id: requestId, correct };
    if (!correct && correctedPlate) body.corrected_plate = correctedPlate;
    return call('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  },
  history: (sessionId) => call(`/history?${new URLSearchParams({ session_id: sessionId, limit: '20' })}`),
};
