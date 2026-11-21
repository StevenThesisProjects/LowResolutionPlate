#!/usr/bin/env python3
"""System Usability Scale (SUS) score from the user-test answers (roadmap week 10).

Input CSV (one row per participant): participant,q1,...,q10 with answers 1-5 (1 = hoàn toàn
không đồng ý, 5 = hoàn toàn đồng ý), using the 10 statements in docs/user_test.md.
Standard SUS scoring (Brooke, 1996): odd items contribute (answer - 1), even items
(5 - answer); the sum times 2.5 gives 0-100.

    python3 scripts/sus_score.py artifacts/sus_results.csv
"""
import csv
import statistics
import sys


def sus(row):
    vals = [int(row[f"q{i}"]) for i in range(1, 11)]
    if any(v < 1 or v > 5 for v in vals):
        raise ValueError(f"{row.get('participant')}: answers must be 1-5")
    return 2.5 * sum((v - 1) if i % 2 == 0 else (5 - v) for i, v in enumerate(vals))


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scores = [(r["participant"], sus(r)) for r in rows]
    for p, s in scores:
        print(f"  {p:12s} {s:5.1f}")
    vals = [s for _, s in scores]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"SUS trung bình {statistics.mean(vals):.1f} ± {sd:.1f} (n = {len(vals)}); "
          "68 được coi là mức trung bình (Sauro, 2011)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/sus_results.csv")
