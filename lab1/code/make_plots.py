#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Построение графиков по результатам экспериментов."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(LAB, "results")
PLOTS = os.path.join(LAB, "plots")
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

with open(os.path.join(RESULTS, "results.json")) as fh:
    R = json.load(fh)


def cpa(rec):
    cycles = rec["basic"]["cycles"]
    acc = rec.get("accesses") or (rec["N"] * rec["repeats"])
    return cycles / acc


# ---------- Этап 4 ----------
s4 = R["stage4"]
w = [r["bytes"] for r in s4]
cpa4 = [cpa(r) for r in s4]
cache_rate = [100 * r["basic"]["cache-misses"] / r["basic"]["cache-references"] for r in s4]
l1_rate = [100 * r["l1"]["L1-dcache-load-misses"] / r["l1"]["L1-dcache-loads"] for r in s4]
llc_rate = [100 * r["llc"]["LLC-load-misses"] / r["llc"]["LLC-loads"] for r in s4]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(w, cpa4, "o-", color="#1f4e79", linewidth=1.8, markersize=6)
for xv, name, col in [(32 * 1024, "L1d 32 КиБ", "green"),
                      (256 * 1024, "L2 256 КиБ", "orange"),
                      (6 * 1024 * 1024, "L3 6 МиБ", "red")]:
    ax.axvline(xv, color=col, linestyle="--", linewidth=1.2, label=name)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Размер рабочего набора W, байт")
ax.set_ylabel("cycles / обращение")
ax.set_title("Этап 4. Стоимость одного обращения к памяти\nв зависимости от размера рабочего набора (double)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage4_cycles_per_access.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(w, l1_rate, "s-", label="L1-dcache miss rate", color="#2e7d32")
ax.plot(w, cache_rate, "^-", label="cache miss rate (LLC)", color="#c62828")
ax.plot(w, llc_rate, "d--", label="LLC-load miss rate", color="#6a1b9a")
ax.set_xscale("log", base=2)
ax.set_xlabel("Размер рабочего набора W, байт")
ax.set_ylabel("Доля промахов, %")
ax.set_title("Этап 4. Доля промахов кэш-памяти\nв зависимости от размера рабочего набора")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage4_miss_rate.png"), dpi=150)
plt.close(fig)

# ---------- Этап 5 ----------
s5 = R["stage5"]
strides = [r["stride"] for r in s5]
cpa5 = [cpa(r) for r in s5]
l1_rate5 = [100 * r["l1"]["L1-dcache-load-misses"] / r["l1"]["L1-dcache-loads"] for r in s5]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(strides, cpa5, "o-", color="#1f4e79", linewidth=1.8, markersize=6, label="cycles/access")
ax.set_xscale("log", base=2)
ax.set_xlabel("stride (шаг в элементах double)")
ax.set_ylabel("cycles / обращение")
ax.set_title("Этап 5. Стоимость обращения к памяти\nв зависимости от шага stride")
ax2 = ax.twinx()
ax2.plot(strides, l1_rate5, "s--", color="#c62828", markersize=5, label="L1 miss rate, %")
ax2.set_ylabel("L1-dcache miss rate, %", color="#c62828")
ax2.grid(False)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage5_stride.png"), dpi=150)
plt.close(fig)

# ---------- Этап 6 ----------
s6 = R["stage6"]
Ns = sorted(set(r["N"] for r in s6))
fl = [cpa(r) for r in s6 if r["type"] == "float"]
db = [cpa(r) for r in s6 if r["type"] == "double"]
import numpy as np
x = np.arange(len(Ns))
width = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.bar(x - width / 2, fl, width, label="float (4 байта)", color="#42a5f5")
ax.bar(x + width / 2, db, width, label="double (8 байт)", color="#1f4e79")
ax.set_xticks(x)
ax.set_xticklabels([f"{n/1e6:.0f}M" for n in Ns])
ax.set_xlabel("Число элементов N")
ax.set_ylabel("cycles / обращение")
ax.set_title("Этап 6. Стоимость обращения для float и double")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage6_types.png"), dpi=150)
plt.close(fig)

# ---------- Этап 7 ----------
s7 = R["stage7"]
rows = s7["data"]
pages = [r["pages"] for r in rows]
tlb_rate = [100 * r["tlb"]["dTLB-load-misses"] / r["tlb"]["dTLB-loads"] for r in rows]
cpa7 = [cpa(r) for r in rows]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.plot(pages, tlb_rate, "o-", color="#c62828", linewidth=1.8, markersize=6)
ax.set_xscale("log")
ax.set_xlabel("Число затронутых страниц памяти (4 КиБ)")
ax.set_ylabel("dTLB miss rate, %")
ax.set_title("Этап 7. Доля промахов TLB\nв зависимости от числа страниц (stride = 512)")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage7_tlb.png"), dpi=150)
plt.close(fig)

# ---------- Этап 8 ----------
s8 = R["stage8"]
modes = ["predictable", "random"]
labels = ["предсказуемые", "непредсказуемые"]
miss_rate8 = [100 * s8[m]["branch-misses"] / s8[m]["branches"] for m in modes]
cycles8 = [s8[m]["cycles"] / 1e9 for m in modes]

fig, ax = plt.subplots(figsize=(7.2, 4.3))
bars = ax.bar(labels, miss_rate8, color=["#2e7d32", "#c62828"], width=0.5)
ax.set_ylabel("Branch miss rate, %")
ax.set_title("Этап 8. Доля ошибочных предсказаний переходов")
for b, v in zip(bars, miss_rate8):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_stage8_branch.png"), dpi=150)
plt.close(fig)

print("Графики сохранены в", PLOTS)
