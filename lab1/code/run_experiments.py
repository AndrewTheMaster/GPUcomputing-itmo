#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Драйвер экспериментов лабораторной работы №1.

Запускает тестовые программы под perf stat, собирает аппаратные метрики
и сохраняет результаты в results/results.json.
"""

import json
import os
import subprocess
import tempfile

LAB = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(LAB, "results")
os.makedirs(RESULTS, exist_ok=True)

DAXPY = os.path.join(LAB, "daxpy")
BRANCH = os.path.join(LAB, "branch_test")
PROFILED = os.path.join(LAB, "profiled_daxpy")

# группы событий, которые гарантированно совместно планируются на Haswell
G_BASIC = ["task-clock", "cycles", "instructions", "cache-references", "cache-misses"]
G_L1 = ["task-clock", "cycles", "instructions", "L1-dcache-loads", "L1-dcache-load-misses"]
G_LLC = ["task-clock", "cycles", "instructions", "LLC-loads", "LLC-load-misses"]
G_TLB = ["task-clock", "cycles", "instructions", "dTLB-loads", "dTLB-load-misses"]
G_BRANCH = ["task-clock", "cycles", "instructions", "branches", "branch-misses"]


def build():
    subprocess.run(["g++", "daxpy.cpp", "-O3", "-std=c++17", "-o", "daxpy"], cwd=LAB, check=True)
    subprocess.run(["g++", "branch_test.cpp", "-O0", "-std=c++17", "-o", "branch_test"], cwd=LAB, check=True)
    subprocess.run(["g++", "function_profiling_example.cpp", "perf_count.cpp", "-O3", "-std=c++17",
                    "-o", "profiled_daxpy"], cwd=LAB, check=True)


def perf(events, prog, repeat=1, timeout=1800):
    """Запускает perf stat -j и возвращает словарь event -> значение."""
    fd, path = tempfile.mkstemp(suffix=".json", dir=RESULTS)
    os.close(fd)
    cmd = ["perf", "stat", "-o", path, "-j"]
    if repeat and repeat > 1:
        cmd += ["-r", str(repeat)]
    cmd += ["-e", ",".join(events), "--"] + prog
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pass
    res = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                d = json.loads(line)
                res[d["event"]] = float(d["counter-value"])
                if "variance" in d:
                    res[d["event"] + "_var"] = float(d["variance"])
    finally:
        os.unlink(path)
    return res


def stage1():
    info = {}
    def cap(key, cmd):
        info[key] = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    cap("uname", "uname -a")
    cap("lscpu", "lscpu")
    cap("lscpu_cache", "lscpu -C")
    cap("pagesize", "getconf PAGESIZE")
    cap("gxx", "g++ --version | head -1")
    cap("perf", "perf --version")
    cap("paranoid", "cat /proc/sys/kernel/perf_event_paranoid")
    cap("hw_events", "perf list hw 2>/dev/null")
    cap("cache_events", "perf list cache 2>/dev/null")
    return info


def stage2():
    args = [DAXPY, "1000000", "100", "1"]
    return {
        "args": args[1:],
        "basic": perf(G_BASIC, args, repeat=5),
        "l1": perf(G_L1, args, repeat=5),
        "llc": perf(G_LLC, args, repeat=5),
    }


def stage3():
    args = [DAXPY, "1000000", "100", "1"]
    out = {}
    out["perf_r10"] = perf(G_BASIC, args, repeat=10)
    out["perf_r10_taskset"] = perf(G_BASIC, ["taskset", "-c", "2"] + args, repeat=10)
    # 10 отдельных запусков для оценки разброса времени
    times = []
    for _ in range(10):
        r = perf(["task-clock", "cycles"], args, repeat=1)
        times.append({"task-clock": r.get("task-clock"), "cycles": r.get("cycles")})
    out["individual"] = times
    return out


def stage4():
    Ns = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
          131072, 262144, 524288, 1048576, 4194304]
    target = 200_000_000
    data = []
    for N in Ns:
        repeats = max(20, round(target / N))
        args = [DAXPY, str(N), str(repeats), "1", "double"]
        rec = {
            "N": N,
            "repeats": repeats,
            "bytes": 2 * N * 8,
            "basic": perf(G_BASIC, args, repeat=3),
            "l1": perf(G_L1, args, repeat=3),
            "llc": perf(G_LLC, args, repeat=3),
        }
        data.append(rec)
        print("  stage4 N=%-9d W=%-10d repeats=%-8d cycles=%s" %
              (N, 2 * N * 8, repeats, rec["basic"].get("cycles")))
    return data


def stage5():
    N = 10_000_000
    strides = [1, 2, 4, 8, 16, 32, 64]
    data = []
    for s in strides:
        # repeats масштабируются так, чтобы число фактических обращений
        # к памяти N/stride*repeats было постоянным (~2*10^8). Иначе
        # инициализация массивов доминировала бы в измерении при большом stride.
        repeats = 20 * s
        args = [DAXPY, str(N), str(repeats), str(s), "double"]
        rec = {
            "N": N, "stride": s, "repeats": repeats,
            "accesses": (N // s) * repeats,
            "basic": perf(G_BASIC, args, repeat=5),
            "l1": perf(G_L1, args, repeat=5),
        }
        data.append(rec)
        print("  stage5 stride=%-3d repeats=%-6d accesses=%-12d cycles=%s" %
              (s, repeats, (N // s) * repeats, rec["basic"].get("cycles")))
    return data


def stage6():
    Ns = [1_000_000, 4_000_000, 16_000_000]
    target = 200_000_000
    data = []
    for N in Ns:
        repeats = max(2, round(target / N))
        for t in ["float", "double"]:
            args = [DAXPY, str(N), str(repeats), "1", t]
            rec = {
                "N": N, "repeats": repeats, "type": t,
                "bytes": 2 * N * (4 if t == "float" else 8),
                "basic": perf(G_BASIC, args, repeat=3),
                "l1": perf(G_L1, args, repeat=3),
                "llc": perf(G_LLC, args, repeat=3),
            }
            data.append(rec)
            print("  stage6 N=%-9d type=%-6s W=%-10d cycles=%s" %
                  (N, t, rec["bytes"], rec["basic"].get("cycles")))
    return data


def stage7():
    stride = 512
    Ns = [100_000, 1_000_000, 10_000_000]
    target_access = 50_000_000
    data = []
    for N in Ns:
        repeats = max(10, round(target_access * stride / N))
        args = [DAXPY, str(N), str(repeats), str(stride), "double"]
        rec = {
            "N": N, "stride": stride, "repeats": repeats,
            "pages": (N * 8) // 4096,
            "accesses": (N // stride) * repeats,
            "tlb": perf(G_TLB, args, repeat=5),
            "basic": perf(G_BASIC, args, repeat=5),
        }
        data.append(rec)
        print("  stage7 N=%-9d pages=%-8d dTLB-misses=%s" %
              (N, (N * 8) // 4096, rec["tlb"].get("dTLB-load-misses")))
    # baseline stride=1
    args = [DAXPY, "10000000", "20", "1", "double"]
    baseline = {"N": 10_000_000, "stride": 1, "repeats": 20,
                "pages": (10_000_000 * 8) // 4096,
                "accesses": 10_000_000 * 20,
                "tlb": perf(G_TLB, args, repeat=5),
                "basic": perf(G_BASIC, args, repeat=5)}
    return {"data": data, "baseline": baseline}


def stage8():
    n = 10_000_000
    repeats = 10
    out = {}
    for mode in ["predictable", "random"]:
        args = [BRANCH, str(n), mode, str(repeats)]
        out[mode] = perf(G_BRANCH, args, repeat=5)
    out["n"] = n
    out["repeats"] = repeats
    return out


def optional_stage():
    out = {}
    modes = {
        "PROFILE_INSTRUCTIONS": "instructions",
        "PROFILE_BRANCHES": "branches",
        "PROFILE_L1_CACHES": "l1",
        "PROFILE_TLB": "tlb",
    }
    for env, key in modes.items():
        envv = dict(os.environ)
        envv[env] = "1"
        p = subprocess.run([PROFILED, "1000000", "10", "1"], capture_output=True, text=True, env=envv)
        out[key] = p.stdout + p.stderr
    return out


def main():
    build()
    results = {}
    print("Stage 1: system info")
    results["stage1"] = stage1()
    print("Stage 2: basic profiling")
    results["stage2"] = stage2()
    print("Stage 3: repeatability")
    results["stage3"] = stage3()
    print("Stage 4: working set")
    results["stage4"] = stage4()
    print("Stage 5: stride")
    results["stage5"] = stage5()
    print("Stage 6: data types")
    results["stage6"] = stage6()
    print("Stage 7: TLB")
    results["stage7"] = stage7()
    print("Stage 8: branch predictor")
    results["stage8"] = stage8()
    print("Optional stage: programmatic counters")
    results["optional"] = optional_stage()

    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("Saved:", os.path.join(RESULTS, "results.json"))


if __name__ == "__main__":
    main()
