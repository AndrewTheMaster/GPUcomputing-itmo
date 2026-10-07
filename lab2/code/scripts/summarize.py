#!/usr/bin/env python3
"""Aggregate validated raw benchmark measurements to means, medians and sample stddev."""
import argparse
import csv
import math
from pathlib import Path
import statistics
import sys

KEYS = ["algorithm", "implementation", "n_a", "n_b", "elements", "bins", "threads", "seed", "pattern", "gpu", "compute_capability", "cuda_runtime", "driver"]
METRICS = ["operation_ms", "device_ms", "end_to_end_ms"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.csv.resolve() == args.out.resolve():
        parser.error("Input and output must be different files")
    if args.out.exists():
        parser.error("Output exists; use a new filename")
    try:
        groups = {}
        with args.csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = set(KEYS + METRICS + ["valid"])
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("Input is not a lab2 raw benchmark CSV")
            for row in reader:
                if row["valid"] != "1":
                    raise ValueError("Input contains invalid measurements")
                key = tuple(row[k] for k in KEYS)
                groups.setdefault(key, []).append(row)
        if not groups:
            raise ValueError("No measurements in CSV")
        result = []
        fields = KEYS + ["samples"]
        for metric in METRICS:
            fields += [f"{metric}_{s}" for s in ("mean", "median", "stddev", "min", "max")]
        for key, rows in groups.items():
            out = dict(zip(KEYS, key)); out["samples"] = len(rows)
            for metric in METRICS:
                vals = [float(r[metric]) for r in rows if r[metric] != ""]
                if any(not math.isfinite(v) or v < 0 for v in vals):
                    raise ValueError("Invalid time in CSV")
                stats = (statistics.mean(vals), statistics.median(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, min(vals), max(vals)) if vals else ("",) * 5
                for suffix, value in zip(("mean", "median", "stddev", "min", "max"), stats):
                    out[f"{metric}_{suffix}"] = value
            result.append(out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(result)
        print(f"{len(result)} configurations -> {args.out}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
