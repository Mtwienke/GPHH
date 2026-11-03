# cross_eval_parallel.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import time

_SIM_BY_CITY = None

def _safe(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.=-]+", "_", str(s))

def init_worker(sim_by_city: dict):
    """
    sim_by_city must be picklable. If your Simulator holds loggers/file handles,
    pass configs instead and construct Simulator inside the worker.
    """
    global _SIM_BY_CITY
    _SIM_BY_CITY = sim_by_city

def eval_pair_and_write(payload: dict):
    """
    payload contains one policy and one testset plus output directory.
    Writes one parquet file and returns (out_path, n_rows).
    """
    p = payload["policy"]
    ts = payload["testset"]
    out_dir = Path(payload["out_dir"])

    sim = _SIM_BY_CITY[ts["test_city"]]

    # build unique filename (one file per policy-testset pair)
    fname = (
        f"cross_eval__P_{_safe(p['policy_city'])}_r={p['policy_r']}_v={p['policy_v']}"
        f"__ON__{_safe(ts['test_city'])}_r={ts['test_r']}_v={ts['test_v']}.parquet"
    )
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, inst in enumerate(ts["instances"]):
        inst_id = inst.get("id", k)
        t0 = time.perf_counter()
        score_total, score_norm = sim.run(
            inst,
            (p["veh_policy"], p["req_policy"]),
        )
        dt = time.perf_counter()-t0
        rows.append({
            "phase": "cross_eval",
            "run_type": "all_vs_all",
            "policy_city": p["policy_city"],
            "policy_r": p["policy_r"],
            "policy_v": p["policy_v"],
            "test_city": ts["test_city"],
            "test_r": ts["test_r"],
            "test_v": ts["test_v"],
            "instance_id": str(inst_id),
            "score_total": float(score_total),
            "score_norm": float(score_norm),
            "runtime_s": float(dt),
        })

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return str(out_path), len(rows)