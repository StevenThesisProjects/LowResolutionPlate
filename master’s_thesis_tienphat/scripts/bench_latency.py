#!/usr/bin/env python3
"""CPU latency of the running API (roadmap weeks 6 and 10: latency_cpu.csv, p50/p95,
5 concurrent requests; MVP criterion 3: p95 <= 1 s for one 5-frame track).

Start the server first, in another terminal:
    cd masters_thesis_tienphat && python3 -m uvicorn mfplate.api:app --port 8000
then:
    python3 scripts/bench_latency.py --url http://127.0.0.1:8000

Scenarios: sequential requests with 5 frames, sequential requests with 1-4 frames, and
5 clients sending at the same time. Client time = full HTTP round trip on this machine;
server time = `latency_ms` in the response (upload parsing to JSON).
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from mfplate.settings import ARTIFACTS_DIR, REPO_ROOT  # noqa: E402

TRACK = "track_14926"


def track_files(n=5):
    d = next((REPO_ROOT / "dataset" / "data" / "train").glob(f"*/*/{TRACK}"))
    return [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in sorted(d.glob("lr-*"))[:n]]


def one(client, url, files):
    t0 = time.perf_counter()
    r = client.post(f"{url}/predict", files=files)
    client_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    j = r.json()
    return client_ms, j["latency_ms"], j["timing"]["model_ms"]


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=40, help="requests per concurrent client")
    args = ap.parse_args()
    rows = []
    with httpx.Client(timeout=60) as client:
        health = client.get(f"{args.url}/health").json()
        if health.get("status") != "ok":
            sys.exit(f"❌ server chưa sẵn sàng: {health}")
        files5 = track_files(5)
        for _ in range(10):                                   # warm-up
            one(client, args.url, files5)
        for i in range(args.requests):
            c, s, m = one(client, args.url, files5)
            rows.append({"scenario": "sequential_5f", "i": i, "client_ms": c, "server_ms": s, "model_ms": m})
        for n in range(1, 5):
            files = track_files(n)
            for i in range(50):
                c, s, m = one(client, args.url, files)
                rows.append({"scenario": f"sequential_{n}f", "i": i, "client_ms": c,
                             "server_ms": s, "model_ms": m})

    def worker(k):
        out = []
        with httpx.Client(timeout=120) as cl:
            for i in range(args.rounds):
                c, s, m = one(cl, args.url, files5)
                out.append({"scenario": f"concurrent_{args.concurrency}x_5f", "i": k * 1000 + i,
                            "client_ms": c, "server_ms": s, "model_ms": m})
        return out

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        for part in ex.map(worker, range(args.concurrency)):
            rows.extend(part)
    wall = time.perf_counter() - t0

    with open(ARTIFACTS_DIR / "latency_cpu.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    summary = {"created": dt.datetime.now().isoformat(timespec="seconds"), "url": args.url,
               "machine": {"platform": platform.platform(), "processor": platform.processor(),
                           "cpu_count": os.cpu_count(), "python": platform.python_version()},
               "scenarios": {}}
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        sel = [r for r in rows if r["scenario"] == sc]
        cl = [r["client_ms"] for r in sel]
        summary["scenarios"][sc] = {
            "requests": len(sel), "client_p50_ms": round(pct(cl, 50), 1),
            "client_p95_ms": round(pct(cl, 95), 1), "client_mean_ms": round(statistics.mean(cl), 1),
            "server_p50_ms": round(pct([r["server_ms"] for r in sel], 50), 1),
            "model_p50_ms": round(pct([r["model_ms"] for r in sel], 50), 1)}
    conc = summary["scenarios"][f"concurrent_{args.concurrency}x_5f"]
    conc["throughput_req_per_s"] = round(args.concurrency * args.rounds / wall, 2)
    p95 = summary["scenarios"]["sequential_5f"]["client_p95_ms"]
    summary["mvp_criterion_p95_le_1000ms"] = {"value_ms": p95, "passed": p95 <= 1000}
    (ARTIFACTS_DIR / "latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    for sc, v in summary["scenarios"].items():
        print(f"  {sc:22s} p50 {v['client_p50_ms']:7.1f} ms  p95 {v['client_p95_ms']:7.1f} ms  "
              f"(model {v['model_p50_ms']:.1f} ms)")
    print(f"📝 {ARTIFACTS_DIR / 'latency_cpu.csv'}\n📝 {ARTIFACTS_DIR / 'latency_summary.json'}")


if __name__ == "__main__":
    main()
