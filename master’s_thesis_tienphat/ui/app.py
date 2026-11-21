"""MF-PlateReader web UI (Gradio). A client of the HTTP API: it never loads the model.

    # terminal 1 (API)
    cd masters_thesis_tienphat && python3 -m uvicorn mfplate.api:app --port 8000
    # terminal 2 (UI)
    cd masters_thesis_tienphat && python3 ui/app.py          # http://127.0.0.1:7860

Environment: MFPLATE_API_URL (default http://127.0.0.1:8000), MFPLATE_API_KEY (if the API
requires one), MFPLATE_UI_HOST / MFPLATE_UI_PORT.
"""
import html
import io
import os
import re
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import httpx
from PIL import Image

API_URL = os.environ.get("MFPLATE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("MFPLATE_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".zip": "application/zip"}
REASONS = {"low_confidence": "độ tin cậy dưới ngưỡng",
           "invalid_layout": "chuỗi không theo mẫu biển Brazil/Mercosur",
           "not_calibrated": "chưa có ngưỡng cho số frame này"}
SCOPE = ("**Phạm vi:** chỉ biển Brazil/Mercosur 1 dòng, 7 ký tự (bộ dữ liệu LRLPR-26). "
         "Không dùng cho biển Việt Nam. Đầu vào là ảnh **đã crop** biển số; kết quả chỉ để tham khảo.")


def _natural(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _paths(files):
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]


def _preview(paths):
    """Images to show next to the result, in the order the API reads them."""
    if len(paths) == 1 and paths[0].lower().endswith(".zip"):
        try:
            z = zipfile.ZipFile(paths[0])
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))
                     and "__MACOSX" not in n]
            lr = [n for n in names if Path(n).name.lower().startswith("lr-")] or names
            return [(Path(n).name, Image.open(io.BytesIO(z.read(n))).convert("RGB"))
                    for n in sorted(lr, key=_natural)]
        except Exception:
            return []
    out = []
    for p in sorted(paths, key=lambda p: _natural(Path(p).name)):
        try:
            out.append((Path(p).name, Image.open(p).convert("RGB")))
        except Exception:
            out.append((Path(p).name, None))
    return out


def _card(j):
    review = j["needs_review"]
    badge = ('<span style="background:#b42318;color:#fff;padding:3px 10px;border-radius:12px">'
             'CẦN KIỂM TRA</span>' if review else
             '<span style="background:#067647;color:#fff;padding:3px 10px;border-radius:12px">'
             'TỰ ĐỘNG</span>')
    reasons = ", ".join(REASONS.get(r, r) for r in j["review_reasons"])
    t = j["triage"]
    thr = "chưa hiệu chỉnh" if t["threshold"] is None else f"{t['threshold']:.4f}"
    ref = ""
    if j.get("reference_plate"):
        ok = j["reference_plate"] == j["plate"]
        ref = (f'<div style="margin-top:6px">Nhãn trong zip: <b style="font-family:monospace">'
               f'{html.escape(j["reference_plate"])}</b> {"✅ đúng" if ok else "❌ sai"}</div>')
    exp = ""
    if j["n_frames"] < 5 and j.get("expected_accuracy") is not None:
        exp = (f'<div style="margin-top:6px">Với {j["n_frames"]} frame, accuracy trung bình trên '
               f'lock-box là {j["expected_accuracy"]:.1f}%.</div>')
    return f"""
<div style="border:1px solid #d0d5dd;border-radius:10px;padding:14px">
  <div style="font-size:12px;opacity:.75">Biển số đọc được ({html.escape(j['layout'] or 'không rõ layout')})</div>
  <div style="font-family:monospace;font-size:40px;font-weight:700;letter-spacing:6px">{html.escape(j['plate'] or '—')}</div>
  <div style="margin:6px 0">{badge}</div>
  <div>Độ tin cậy: <b>{100 * j['confidence']:.2f}%</b> · xác suất cả chuỗi: {100 * j['sequence_prob']:.2f}%</div>
  <div style="font-size:13px;opacity:.85">Phân loại theo <code>{t['score_name']}</code> = {t['score']:.4f}, ngưỡng cho {j['n_frames']} frame: {thr}{(' · lý do: ' + html.escape(reasons)) if reasons else ''}</div>
  {exp}{ref}
  <div style="font-size:12px;opacity:.7;margin-top:6px">{j['n_frames']} frame · decoder {j['decoder']} · {j['latency_ms']:.0f} ms · {html.escape(j['model_version'])}</div>
</div>"""


def predict(files, decoder, session_id):
    paths = _paths(files)
    empty = (None, [], [], [], "", None, None, gr.update(visible=False))
    if not paths:
        return ("<b>Hãy chọn 1–5 ảnh crop biển số, hoặc 1 file zip của một track.</b>",) + empty[1:]
    multipart = [("files", (Path(p).name, Path(p).read_bytes(),
                            MIME.get(Path(p).suffix.lower(), "application/octet-stream")))
                 for p in paths]
    params = {"session_id": session_id} if session_id else {}
    if decoder != "mặc định":
        params["decoder"] = decoder
    try:
        r = httpx.post(f"{API_URL}/predict", files=multipart, params=params, headers=HEADERS,
                       timeout=60)
    except httpx.HTTPError as e:
        return (f"<b>Không kết nối được API ({API_URL}): {html.escape(str(e))}</b>",) + empty[1:]
    j = r.json()
    if r.status_code != 200:
        err = j.get("error", {})
        return (f"<b>Lỗi {r.status_code} · {html.escape(err.get('code', ''))}</b>: "
                f"{html.escape(err.get('message', ''))}",) + empty[1:]
    preview = _preview(paths)
    gallery = []
    for f in j["frames"]:
        img = preview[f["index"]][1] if f["index"] < len(preview) else None
        if img is not None:
            gallery.append((img, f"{f['name']} · trọng số {f['weight']:.2f} · độ nét {f['sharpness']:.0f}"))
    top = [[k + 1, c["plate"], f"{100 * c['prob']:.3f}%"] for k, c in enumerate(j["top3"])]
    chars = [[k + 1, ch, f"{100 * c:.2f}%"] for k, (ch, c) in enumerate(zip(j["plate"], j["char_conf"]))]
    warn = "\n".join(f"- ⚠️ {w}" for w in j["warnings"])
    return (_card(j), gallery, top, chars, warn, j, j["request_id"], gr.update(visible=True))


def send_feedback(request_id, correct, corrected):
    if not request_id:
        return "Chưa có kết quả để phản hồi."
    body = {"request_id": request_id, "correct": correct}
    if not correct and corrected and corrected.strip():
        body["corrected_plate"] = corrected.strip().upper()
    try:
        r = httpx.post(f"{API_URL}/feedback", json=body, headers=HEADERS, timeout=15)
    except httpx.HTTPError as e:
        return f"Không gửi được phản hồi: {e}"
    if r.status_code != 200:
        return f"Lỗi {r.status_code}: {r.json().get('error', {}).get('message', '')}"
    return "Đã ghi nhận: kết quả đúng." if correct else "Đã ghi nhận: kết quả sai."


def history(session_id):
    if not session_id:            # no session yet: never list other users' predictions
        return []
    try:
        r = httpx.get(f"{API_URL}/history", params={"session_id": session_id, "limit": 50},
                      headers=HEADERS, timeout=15)
        rows = r.json() if r.status_code == 200 else []
    except httpx.HTTPError:
        rows = []
    fb = {True: "đúng", False: "sai", None: ""}
    return [[h["created_at"][11:19], h["plate"], f"{100 * h['confidence']:.1f}%",
             "cần kiểm tra" if h["needs_review"] else "tự động", h["n_frames"],
             fb[h["feedback_correct"]] + (f" → {h['corrected_plate']}" if h["corrected_plate"] else "")]
            for h in rows]


def model_card():
    try:
        m = httpx.get(f"{API_URL}/model", headers=HEADERS, timeout=10).json()
        thr = ", ".join(f"{n} frame: {v:.4f}" if v is not None else f"{n} frame: —"
                        for n, v in m["thresholds_by_frames"].items())
        acc = ", ".join(f"{n}: {v:.1f}%" for n, v in m["expected_accuracy_by_frames"].items())
        return (f"- Model: `{m['model_version']}` ({m['parameters']:,} tham số), SHA-256 "
                f"{'đã kiểm' if m['sha256_verified'] else 'CHƯA KHỚP'}\n"
                f"- Decoder mặc định: `{m['default_decoder']}`; điểm phân loại: `{m['triage_score']}`\n"
                f"- Ngưỡng \"cần kiểm tra\": {thr}\n- Accuracy lock-box theo số frame: {acc}\n"
                f"- Cách chọn checkpoint: {m['selection_rule']}")
    except Exception as e:
        return f"Không đọc được /model từ {API_URL}: {e}"


def build():
    samples = sorted(str(p) for p in SAMPLES.glob("*.zip"))
    with gr.Blocks(title="MF-PlateReader") as demo:
        session = gr.State()
        request_id = gr.State()
        gr.Markdown("# MF-PlateReader\nĐọc biển số từ 1–5 ảnh crop độ phân giải thấp của cùng một xe.")
        gr.Markdown(SCOPE)
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Ảnh crop (jpg/png, tối đa 5) hoặc 1 file zip của track",
                                file_count="multiple", file_types=[".jpg", ".jpeg", ".png", ".zip"],
                                type="filepath")
                decoder = gr.Radio(["mặc định", "greedy", "beam", "constrained"], value="mặc định",
                                   label="Bộ giải mã")
                run = gr.Button("Đọc biển số", variant="primary")
                if samples:
                    gr.Examples(examples=[[[s]] for s in samples], inputs=[files],
                                label="Track mẫu (val 999)")
            with gr.Column(scale=1):
                card = gr.HTML()
                warn = gr.Markdown()
                with gr.Group(visible=False) as fb_box:
                    gr.Markdown("Kết quả này đúng không?")
                    with gr.Row():
                        ok_btn = gr.Button("✅ Đúng")
                        bad_btn = gr.Button("❌ Sai")
                    corrected = gr.Textbox(label="Biển số đúng (nếu sai)", max_lines=1)
                    fb_status = gr.Markdown()
        gallery = gr.Gallery(label="Các frame (trọng số fusion, độ nét)", columns=5, height=220,
                             object_fit="contain")
        with gr.Row():
            top = gr.Dataframe(headers=["#", "Biển số", "Xác suất"], label="Top-3 phương án",
                               interactive=False)
            chars = gr.Dataframe(headers=["Vị trí", "Ký tự", "Độ tin cậy"],
                                 label="Độ tin cậy từng ký tự", interactive=False)
        with gr.Accordion("Lịch sử trong phiên", open=False):
            hist = gr.Dataframe(headers=["Giờ (UTC)", "Biển số", "Tin cậy", "Phân loại", "Frame",
                                         "Phản hồi"], interactive=False)
            refresh = gr.Button("Tải lại lịch sử", size="sm")
        with gr.Accordion("JSON trả về từ API", open=False):
            raw = gr.JSON()
        with gr.Accordion("Thông tin model", open=False):
            info = gr.Markdown()

        demo.load(lambda: uuid.uuid4().hex, outputs=session).then(model_card, outputs=info)
        run.click(predict, [files, decoder, session],
                  [card, gallery, top, chars, warn, raw, request_id, fb_box]).then(
            history, session, hist)
        ok_btn.click(lambda rid: send_feedback(rid, True, None), request_id, fb_status).then(
            history, session, hist)
        bad_btn.click(lambda rid, c: send_feedback(rid, False, c), [request_id, corrected],
                      fb_status).then(history, session, hist)
        refresh.click(history, session, hist)
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("MFPLATE_UI_HOST", "127.0.0.1"),
                   server_port=int(os.environ.get("MFPLATE_UI_PORT", "7860")),
                   theme=gr.themes.Soft())
