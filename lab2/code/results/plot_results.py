#!/usr/bin/env python3
"""Build report figures from the summarized series measurements."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = row[key]
    return float(v) if v not in ("", None) else None


series = load(HERE / "series_summary.csv")
repeats = load(HERE / "repeats_summary.csv")

# ---- Figure 1: median end_to_end time vs N (student B=256, CPU, Thrust) ----
by = defaultdict(dict)
for r in series:
    n = int(r["elements"])
    impl = r["implementation"]
    b = int(r["threads"])
    if impl == "student":
        if b != 256:
            continue
    by[n][impl] = fnum(r, "end_to_end_ms_median")

ns = sorted(by)
fig, ax = plt.subplots(figsize=(8, 5))
for impl, label in (("cpu", "CPU"), ("student", "Собственный CUDA (B=256)"), ("thrust", "Thrust")):
    ys = [by[n].get(impl) for n in ns]
    ax.plot(ns, ys, marker="o", label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("Размер массива N, элементов")
ax.set_ylabel("Медиана end_to_end_ms, мс")
ax.set_title("Время histogram от размера N (медиана, 10 повторов)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "time_vs_n.png", dpi=150)
plt.close(fig)

# ---- Figure 2: student median time vs block size B at N=1048576 ----
blocks = sorted({int(r["threads"]) for r in series if r["implementation"] == "student"})
target_n = max(ns)
op, e2e = [], []
for b in blocks:
    row = next(r for r in series if r["implementation"] == "student"
               and int(r["threads"]) == b and int(r["elements"]) == target_n)
    op.append(fnum(row, "operation_ms_median"))
    e2e.append(fnum(row, "end_to_end_ms_median"))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(blocks, op, marker="s", label="operation_ms")
ax.plot(blocks, e2e, marker="o", label="end_to_end_ms")
ax.set_xlabel("Число потоков в блоке B")
ax.set_ylabel("Медиана времени, мс")
ax.set_title(f"Собственная реализация histogram, N={target_n}")
ax.set_xticks(blocks)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "time_vs_block.png", dpi=150)
plt.close(fig)

# ---- Figure 3: median end_to_end time vs N, all block sizes ----
fig, ax = plt.subplots(figsize=(8, 5))
for b in blocks:
    xs, ys = [], []
    for n in ns:
        row = next(r for r in series if r["implementation"] == "student"
                   and int(r["threads"]) == b and int(r["elements"]) == n)
        xs.append(n)
        ys.append(fnum(row, "end_to_end_ms_median"))
    ax.plot(xs, ys, marker="o", label=f"B={b}")
ax.set_xscale("log", base=2)
ax.set_xlabel("Размер массива N, элементов")
ax.set_ylabel("Медиана end_to_end_ms, мс")
ax.set_title("Собственная реализация histogram: время от N для разных B")
ax.grid(True, which="both", alpha=0.3)
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUT / "student_time_vs_n_blocks.png", dpi=150)
plt.close(fig)

print("figures written to", OUT)
