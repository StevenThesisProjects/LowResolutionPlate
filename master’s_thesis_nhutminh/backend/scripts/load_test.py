"""Load test đọc/ghi đồng thời trên stack đang chạy (localhost:8000).

1 batch 40 track (ghi liên tục) + 5 client POST /predictions + 5 client đọc liên tục.
Chạy từ master’s_thesis_nhutminh/:
    backend/.venv/bin/python backend/scripts/load_test.py .
"""
import glob, io, json, statistics as st, sys, threading, time, zipfile
import httpx
B = "http://localhost:8000/api/v1"
root = sys.argv[1]
val = json.load(open(f"{root}/dataset/val_tracks.json"))
paths = {p.split('/')[-2]: p.rsplit('/', 1)[0] for p in glob.glob(f"{root}/dataset/data/train/Scenario-B/*/*/annotations.json")}
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    for t in val[:40]:
        for f in glob.glob(paths[t] + "/*"):
            z.write(f, f"{t}/{f.rsplit('/', 1)[1]}")
c = httpx.Client(timeout=120)
r = c.post(f"{B}/batches", files={"file": ("val40.zip", buf.getvalue(), "application/zip")})
bid = r.json()["id"]; print("batch", r.status_code)
lat = {"predict": [], "list": [], "get_batch": []}; errors = []
def predict(i):
    t = val[100 + i]
    files = [("files", (f"lr-00{k}.jpg", open(f"{paths[t]}/lr-00{k}.jpg", "rb").read(), "image/jpeg")) for k in range(1, 6)]
    t0 = time.time(); r = httpx.post(f"{B}/predictions", files=files, data={"track_id": t}, timeout=120)
    (lat["predict"] if r.status_code == 201 else errors).append(time.time() - t0 if r.status_code == 201 else r.status_code)
def reader():
    with httpx.Client(timeout=30) as rc:
        for _ in range(40):
            t0 = time.time(); r = rc.get(f"{B}/predictions", params={"status": "needs_review", "page_size": 50})
            (lat["list"] if r.status_code == 200 else errors).append(time.time() - t0 if r.status_code == 200 else r.status_code)
            t0 = time.time(); r = rc.get(f"{B}/batches/{bid}")
            (lat["get_batch"] if r.status_code == 200 else errors).append(time.time() - t0 if r.status_code == 200 else r.status_code)
            time.sleep(0.25)
ths = [threading.Thread(target=predict, args=(i,)) for i in range(5)] + [threading.Thread(target=reader) for _ in range(5)]
[t.start() for t in ths]; [t.join() for t in ths]
while c.get(f"{B}/batches/{bid}").json()["status"] in ("queued", "running"): time.sleep(2)
b = c.get(f"{B}/batches/{bid}").json()
print("batch", b["status"], b["processed_tracks"], "acc", b["accuracy"])
for k, v in lat.items():
    v = sorted(v); print(f"{k:10s} n={len(v):3d} p50={st.median(v)*1000:7.0f}ms p95={v[int(0.95*(len(v)-1))]*1000:7.0f}ms max={v[-1]*1000:7.0f}ms")
print("errors", errors)
