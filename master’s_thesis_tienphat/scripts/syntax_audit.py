#!/usr/bin/env python3
"""How many errors could a decoder fix? Error audit of the frozen recipe on val 999.

Replaces roadmap week 1 (error_syntax_audit.md: share of errors that violate the plate
syntax = the ceiling of SC-CTC; top-10 confusion pairs by layout), on the 4 lock-box
recipe models (last epoch) instead of run #13.

For every error of the greedy decoder it records
  * the type: wrong length / 7 chars but no valid pattern / valid pattern of the OTHER
    layout / valid pattern of the right layout (a pure character confusion);
  * whether p(ground truth) > p(prediction) under the model's own CTC distribution. Only
    such errors can be fixed by ANY decoder, so their share bounds what decoding can buy;
  * the number of wrong characters and which characters were confused.

    python3 scripts/syntax_audit.py
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mfplate.decoding import ctc_log_prob, greedy, ids_of, layout_of  # noqa: E402
from mfplate.settings import ARTIFACTS_DIR, THESIS_DIR  # noqa: E402

CACHE = THESIS_DIR / "var" / "cache"
RUNS = ["combo3_s42", "combo3_s43", "combo3_s7", "combo3_s2026"]


def classify(pred, gt, gt_layout):
    if len(pred) != len(gt):
        return "wrong_length"
    lay = layout_of(pred)
    if lay is None:
        return "invalid_pattern"
    return "other_layout" if lay != gt_layout else "same_layout_confusion"


def main(split="val"):
    meta = np.load(CACHE / f"{split}__meta.npz")
    labels, layouts = meta["labels"], meta["layouts"]
    per_run, pairs, positions, nwrong = {}, collections.Counter(), collections.Counter(), collections.Counter()
    pairs_by_layout = {"Mercosur": collections.Counter(), "Brazilian": collections.Counter()}
    for run in RUNS:
        lp = np.load(CACHE / f"{split}__{run}_ep30.npz")["log_probs"]
        types = collections.Counter()
        gt_more_likely = 0
        n_err = 0
        for x, gt, lay in zip(lp, labels, layouts):
            pred, _ = greedy(x)
            if pred == gt:
                continue
            n_err += 1
            types[classify(pred, gt, lay)] += 1
            if ctc_log_prob(x, ids_of(gt)) > ctc_log_prob(x, ids_of(pred)):
                gt_more_likely += 1
            if len(pred) == len(gt):
                wrong = [(i, g, p) for i, (g, p) in enumerate(zip(gt, pred)) if g != p]
                nwrong[len(wrong)] += 1
                for i, g, p in wrong:
                    key = "↔".join(sorted((g, p)))
                    pairs[key] += 1
                    pairs_by_layout[lay][f"{g}→{p}"] += 1
                    positions[i + 1] += 1
        per_run[run] = {
            "errors": n_err, "accuracy": round(100 * (1 - n_err / len(labels)), 2),
            "types": dict(types),
            "syntax_violations": types["wrong_length"] + types["invalid_pattern"],
            "syntax_violation_share": round(
                (types["wrong_length"] + types["invalid_pattern"]) / max(n_err, 1), 4),
            "other_layout": types["other_layout"],
            "gt_more_probable_than_prediction": gt_more_likely,
        }
    total_err = sum(r["errors"] for r in per_run.values())
    total_viol = sum(r["syntax_violations"] for r in per_run.values())
    total_fixable = sum(r["gt_more_probable_than_prediction"] for r in per_run.values())
    same_len = sum(nwrong.values())
    report = {
        "split": "splits/val_tracks.json (999 tracks)", "runs": RUNS, "decoder": "greedy",
        "per_run": per_run,
        "pooled": {
            "errors": total_err, "syntax_violations": total_viol,
            "syntax_violation_share": round(total_viol / total_err, 4),
            "gt_more_probable": total_fixable,
            "gt_more_probable_share": round(total_fixable / total_err, 4),
            "errors_same_length": same_len,
            "wrong_chars_distribution": {str(k): v for k, v in sorted(nwrong.items())},
            "one_wrong_char_share_of_all_errors": round(nwrong[1] / total_err, 4),
            "wrong_char_by_position": {str(k): v for k, v in sorted(positions.items())},
        },
        "top10_pairs": pairs.most_common(10),
        "top10_by_layout": {k: v.most_common(10) for k, v in pairs_by_layout.items()},
    }
    out_json = ARTIFACTS_DIR / "error_syntax_audit.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    p = report["pooled"]
    lines = [
        "# Audit lỗi cú pháp (sinh tự động bởi `scripts/syntax_audit.py`)", "",
        "Model: cấu hình chốt #50, 4 run lock-box (seed 42, 43, 7, 2026), epoch cuối, "
        "decoder greedy. Tập: val 999 (`splits/val_tracks.json`), các model này chưa từng "
        "train trên tập này.", "",
        "| Run | Accuracy | Số lỗi | Sai độ dài | 7 ký tự nhưng sai mẫu | Đúng mẫu của layout kia "
        "| Nhầm ký tự trong đúng layout | p(GT) > p(dự đoán) |",
        "|---|---|---|---|---|---|---|---|"]
    for run, r in per_run.items():
        t = r["types"]
        lines.append(f"| {run} | {r['accuracy']:.2f} | {r['errors']} | {t.get('wrong_length', 0)} "
                     f"| {t.get('invalid_pattern', 0)} | {t.get('other_layout', 0)} "
                     f"| {t.get('same_layout_confusion', 0)} "
                     f"| {r['gt_more_probable_than_prediction']} |")
    lines += [
        "", f"- Gộp 4 run: {p['errors']} lỗi; **{p['syntax_violations']} lỗi vi phạm cú pháp "
        f"({100 * p['syntax_violation_share']:.1f}%)**. Đây là trần lợi ích của decode có ràng "
        "buộc cú pháp (SC-CTC).",
        f"- Ở {p['gt_more_probable']} lỗi ({100 * p['gt_more_probable_share']:.1f}%), chuỗi đúng "
        "có xác suất CTC cao hơn chuỗi greedy chọn. Chỉ những lỗi này mới có thể sửa bằng bất kỳ "
        "cách decode nào.",
        f"- {100 * p['one_wrong_char_share_of_all_errors']:.1f}% lỗi chỉ sai đúng 1 ký tự. Phân bố "
        f"số ký tự sai (lỗi cùng độ dài): {p['wrong_chars_distribution']}.",
        f"- Vị trí ký tự sai (1–7): {p['wrong_char_by_position']}.", "",
        "## Top-10 cặp nhầm (gộp 4 run, không phân biệt chiều)", "",
        "| Cặp | Số lần |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["top10_pairs"]]
    for lay, items in report["top10_by_layout"].items():
        lines += ["", f"## Top-10 nhầm theo layout: {lay} (đúng → đọc thành)", "",
                  "| Nhầm | Số lần |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in items]
    out_md = ARTIFACTS_DIR / "error_syntax_audit.md"
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"📝 {out_md}\n📝 {out_json}")


if __name__ == "__main__":
    main()
