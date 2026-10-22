"""Parity: đường tiền xử lý của API vs pipeline offline (MultiFrameDataset) trên 999 track val.

Chạy từ master’s_thesis_nhutminh/ (không cần DB):
    backend/.venv/bin/python backend/scripts/parity_check.py
In ra 1 dòng JSON (dùng cho parity_report.json ở Tuần 7).
"""
import json, os, sys, time
import cv2, torch
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
from backend.app.core.config import Settings
from backend.app.services.inference import PlateRecognizer
from src.data.dataset import MultiFrameDataset
from src.utils.postprocess import decode_batch

rec = PlateRecognizer(Settings(torch_threads=4))
cfg = rec.config
ds = MultiFrameDataset(cfg.DATA_ROOT, mode="val", split_ratio=cfg.SPLIT_RATIO, img_height=cfg.IMG_HEIGHT,
                       img_width=cfg.IMG_WIDTH, char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE,
                       seed=cfg.SEED, num_frames=cfg.NUM_FRAMES)
loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=MultiFrameDataset.collate_fn, num_workers=0)
paths = {s["track_id"]: s["paths"] for s in ds.samples}
offline, api, labels = {}, {}, {}
t0 = time.time()
with torch.inference_mode():
    for batch in loader:
        images, _, _, lab, tids = batch[:5]
        for tid, text in zip(tids, decode_batch(rec.model(images), cfg.IDX2CHAR, mode="greedy")):
            offline[tid] = text[0]
        labels.update(zip(tids, lab))
        frames = [[cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths[t]] for t in tids]
        res, _ = rec.predict(frames, "greedy", top_k=1)
        api.update({t: r.plate for t, r in zip(tids, res)})
same = sum(offline[t] == api[t] for t in offline)
acc = lambda d: 100 * sum(d[t] == labels[t] for t in d) / len(d)
out = {"tracks": len(offline), "decode": "greedy", "identical": same, "mismatch": len(offline) - same,
       "acc_offline": round(acc(offline), 2), "acc_api": round(acc(api), 2),
       "checkpoint": rec.checkpoint_name, "sha256": rec.checkpoint_sha256, "seconds": round(time.time() - t0)}
print(json.dumps(out))
