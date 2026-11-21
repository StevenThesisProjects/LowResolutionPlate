import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the UI on :5173 and forwards API calls to the FastAPI server on :8000.
// `npm run build` writes dist/, which FastAPI serves at / (same origin, no CORS needed).
const API = process.env.MFPLATE_API_URL || 'http://127.0.0.1:8000';
const apiPaths = ['/predict', '/feedback', '/history', '/stats', '/model', '/health', '/samples'];

export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 800 },
  server: { proxy: Object.fromEntries(apiPaths.map((p) => [p, API])) },
});
