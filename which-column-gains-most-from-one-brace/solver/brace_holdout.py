# -*- coding: utf-8 -*-
"""Untouched two-condition parameter holdout for redesign v4.

The conditions were frozen in reference/redesign_registration_v4.md
before execution. Each JSONL line is flushed for resumability.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

from brace_batches import PHASES
from brace_study import NAMES, ROOT, solve

OUT = ROOT / "data" / "holdout"
CONDITIONS = ((0.00075, 0.475, 0.75), (0.00125, 0.525, 1.50))


def jobs():
    for amp, height, factor in CONDITIONS:
        for name in NAMES:
            yield (name, amp, 0.5, 0.0, "one_wave")
            for shape in ("one_wave", "two_wave"):
                yield (name, amp, height, factor, shape)


def key(job):
    n, amp, height, factor, shape = job
    return f"{n}|{amp:.6f}|{height:.3f}|{factor:.2f}|{shape}"


def execute(job, settings):
    name, amp, height, factor, shape = job
    return key(job), solve(name, n=settings["n"], amp=amp, height=height,
                           brace_factor=factor, shape=shape,
                           steps=settings["steps"], tol=settings["tol"])


def load(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["key"]] = row["result"]
    return done


def run(phase, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{phase}.jsonl"
    done = load(path)
    todo = [j for j in jobs() if key(j) not in done]
    print(phase, "done", len(done), "remaining", len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open(
            "a", encoding="utf-8") as f:
        futures = [pool.submit(execute, j, PHASES[phase]) for j in todo]
        for i, future in enumerate(as_completed(futures), 1):
            k, result = future.result()
            f.write(json.dumps({"key": k, "result": result}) + "\n")
            f.flush()
            if i % 6 == 0:
                print(phase, i, "/", len(todo), flush=True)
    if len(load(path)) != 24:
        raise AssertionError(f"{phase} incomplete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=PHASES)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = p.parse_args()
    run(args.phase, args.workers)
