# -*- coding: utf-8 -*-
"""Resumable registered calculation batches, one physical grid per phase.

Each JSONL line is one completed solve and is flushed immediately. A restarted
phase skips completed keys. Controls vary one numerical setting at a time.
Run with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 on Windows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from brace_study import NAMES, ROOT, solve

DATA = ROOT / "data"
AMPS = (1/2000, 1/1000, 1/500)
HEIGHTS = (0.45, 0.50, 0.55)
FACTORS = (0.5, 1.0, 2.0)
SHAPES = ("one_wave", "two_wave")
PHASES = {
    "production": dict(n=80, steps=2400, tol=1e-8),
    "mesh": dict(n=40, steps=2400, tol=1e-8),
    "steps": dict(n=80, steps=1200, tol=1e-8),
    "tol_lo": dict(n=80, steps=2400, tol=1e-7),
    "tol_hi": dict(n=80, steps=2400, tol=1e-9),
}


def make_jobs():
    # All 216 braced cases plus 24 bare references including two-wave controls.
    for amp in AMPS:
        for name in NAMES:
            for shape in SHAPES:
                yield (name, amp, 0.5, 0.0, shape)
        for height in HEIGHTS:
            for factor in FACTORS:
                for name in NAMES:
                    for shape in SHAPES:
                        yield (name, amp, height, factor, shape)


def key(job):
    name, amp, height, factor, shape = job
    return f"{name}|{amp:.7f}|{height:.2f}|{factor:.1f}|{shape}"


def execute(job, settings):
    name, amp, height, factor, shape = job
    result = solve(name, n=settings["n"], amp=amp, height=height,
                   brace_factor=factor, shape=shape, steps=settings["steps"],
                   tol=settings["tol"])
    return key(job), result


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row["result"]
    return done


def run(phase: str, workers: int) -> int:
    settings = PHASES[phase]
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{phase}.jsonl"
    done = load_done(path)
    jobs = [j for j in make_jobs() if key(j) not in done]
    print(f"phase={phase} settings={settings} completed={len(done)} "
          f"remaining={len(jobs)} workers={workers}", flush=True)
    if not jobs:
        return 0
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open(
            "a", encoding="utf-8") as f:
        futures = [pool.submit(execute, j, settings) for j in jobs]
        for idx, task in enumerate(as_completed(futures), 1):
            k, result = task.result()
            f.write(json.dumps({"key": k, "result": result},
                               separators=(",", ":")) + "\n")
            f.flush()
            if idx % 24 == 0 or idx == len(jobs):
                print(f"phase={phase} {idx}/{len(jobs)}", flush=True)
    if len(load_done(path)) != 240:
        raise AssertionError("phase incomplete after all futures")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=PHASES)
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    a = p.parse_args()
    return run(a.phase, a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
