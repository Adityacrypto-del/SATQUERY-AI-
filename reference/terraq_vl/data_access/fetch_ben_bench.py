"""Fetch the 1,082 BigEarthNet.txt bench patches (all 12 S2 bands) via LMDB range reads."""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load as st_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from httprange import HTTPRangeFile, RemoteLMDB  # noqa: E402

URL = "https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/resolve/main/BENv2.lmdb/data.mdb"
OUT = "D:/Academics/SIH/SatQuery/_data/bigearthnet_txt/bench_patches"
os.makedirs(OUT, exist_ok=True)
pids = sorted(set(pq.read_table("D:/Academics/SIH/SatQuery/_data/bigearthnet_txt/bench.parquet",
                                columns=["patch_id"])["patch_id"].to_pylist()))
todo = [p for p in pids if not os.path.exists(f"{OUT}/{p}.npz")]
print("bench patches", len(pids), "to fetch", len(todo), flush=True)

local = threading.local()


def db():
    if not hasattr(local, "db"):
        local.f = HTTPRangeFile(URL)
        local.db = RemoteLMDB(local.f)
    return local.db


def fetch(pid):
    for attempt in range(4):
        try:
            raw = db().get(pid.encode())
            if raw is None:
                return pid, "MISSING"
            np.savez_compressed(f"{OUT}/{pid}.npz", **st_load(raw))
            return pid, "ok"
        except Exception as exc:  # network hiccup: retry
            err = exc
            time.sleep(2 * (attempt + 1))
    return pid, f"ERROR {err}"


t = time.time()
bad = []
with ThreadPoolExecutor(12) as ex:
    for i, fut in enumerate(as_completed([ex.submit(fetch, p) for p in todo]), 1):
        pid, status = fut.result()
        if status != "ok":
            bad.append((pid, status))
        if i % 100 == 0:
            print(f"{i}/{len(todo)} {round(time.time() - t)}s", flush=True)
print("done", len(todo) - len(bad), "ok;", "failed", bad, round(time.time() - t), "s", flush=True)
