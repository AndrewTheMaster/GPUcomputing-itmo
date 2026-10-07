#!/usr/bin/env python3
"""Run one compiled benchmark over N and block sizes. Uses only Python's standard library."""
import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 16384, 65536, 1048576])
    parser.add_argument("--blocks", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--pattern", default="random")
    args = parser.parse_args()
    if not args.exe.is_file():
        parser.error(f"Benchmark not found: {args.exe}")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}. Use a new filename to avoid mixing runs.")
    if any(n < 0 for n in args.sizes) or not 1 <= args.repeat <= 10000:
        parser.error("Sizes must be non-negative; repeat must be 1..10000")
    if any(b not in (32, 64, 128, 256, 512) for b in args.blocks):
        parser.error("Block sizes must be 32,64,128,256,512")
    try:
        for n in args.sizes:
            for block in args.blocks:
                cmd = [str(args.exe.resolve()), "--n", str(n), "--threads", str(block),
                       "--repeat", str(args.repeat), "--bins", str(args.bins),
                       "--pattern", args.pattern, "--impl", "all", "--csv", str(args.out)]
                print(f"N={n}, B={block}", flush=True)
                subprocess.run(cmd, check=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Series stopped: {exc}. Completed experiments may remain in {args.out}.", file=sys.stderr)
        return 1
    print(f"Series completed: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
