"""Print the Task 4 table from the raw CSVs, means +- spread over the three seeds.

  .venv/Scripts/python.exe summarize.py            # everything
  .venv/Scripts/python.exe summarize.py --markdown # markdown table for REPORT.md
"""
import argparse, csv, sys
import numpy as np

SETS = [("val", "group val 29"), ("cam", "camera-like 29"), ("hist", "last-year 120")]

def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    order = []
    for r in rows:
        if r["variant"] not in order:
            order.append(r["variant"])
    return rows, order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="*", default=["results_task4_raw.csv", "results_combos_raw.csv"])
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--state", default="best", choices=["best", "last", "both"])
    args = ap.parse_args()
    rows, order = load(args.csv)
    states = ["best", "last"] if args.state == "both" else [args.state]
    if args.markdown:
        print("| variant | train images | " + " | ".join(n for _, n in SETS) + " |")
        print("|---|---:|" + "---:|" * len(SETS))
    for variant in order:
        for state in states:
            sel = [r for r in rows if r["variant"] == variant and r["state"] == state]
            if not sel:
                continue
            cells = []
            for key, _ in SETS:
                v = [float(r[f"{key}_acc"]) for r in sel]
                cells.append(f"{np.mean(v)*100:4.1f} ± {np.std(v)*100:3.1f}")
            n = sel[0]["n_train"]
            if args.markdown:
                print(f"| {variant} | {n} | " + " | ".join(cells) + " |")
            else:
                print(f"{variant[:46]:48s} {state:4s} n={n:>3} " +
                      "  ".join(f"{n2}={c}" for c, (_, n2) in zip(cells, SETS)))
    # a note on how many runs the noise estimate comes from
    if not args.markdown:
        print(f"\n{len(rows)} runs from {len(order)} variants x seeds {[r['seed'] for r in rows[:3]]}")

if __name__ == "__main__":
    sys.exit(main())
