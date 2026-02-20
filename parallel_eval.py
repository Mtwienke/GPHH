# parallel_eval.py
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd

_WORKER_SIM = 2
_WORKER_travel_metrics = None
_WORKER_INSTANCES = 2

def _init_worker(simulator, travel_metrics, instances):
    global _WORKER_SIM, _WORKER_travel_metrics, _WORKER_INSTANCES
    # Optional: disable logging inside workers
    #try:
    #    simulator.log = None
    #except Exception:
    #    pass

    _WORKER_SIM = simulator
    _WORKER_travel_metrics = travel_metrics
    _WORKER_INSTANCES = instances
    
def _eval_chunk_and_write(payload):
    """
    payload = {
        "tasks": [(idx, veh_policy, req_policy), ...],
        "base_dir": ".../results/city_vxx_rxx",
        "phase": "train",
        "gen": 7,
        "shard_id": 3,
    }
    Returns: [(idx, fitness), ...]
    Also writes parquet file for the shard.
    """
    tasks = payload["tasks"]
    base_dir = payload["base_dir"]
    phase = payload["phase"]
    gen = payload.get("gen", None)
    shard_id = payload["shard_id"]

    if not _WORKER_INSTANCES:
        raise ValueError("Worker received empty instances list.")

    # output path
    if gen is None:
        out_path = f"{base_dir}/runs/phase={phase}/shard={shard_id:03d}.parquet"
    else:
        out_path = f"{base_dir}/runs/phase={phase}/gen={gen:04d}/shard={shard_id:03d}.parquet"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    results = []

    n_inst = len(_WORKER_INSTANCES)

    for idx, veh_policy, req_policy in tasks:
        total_sum = 0.0
        norm_sum = 0.0

        for k, inst in enumerate(_WORKER_INSTANCES):
            # Expecting Sim.run returns (score_total, score_norm)
            out = _WORKER_SIM.run(
                inst,
                (veh_policy, req_policy),
                travel_metrics=_WORKER_travel_metrics
            )

            # --- Adapt if your Sim.run returns different structure ---
            # If it returns a tuple:
            score_total, score_norm = out
            # If it returns a dict like {"total":..., "norm":...}, use:
            # score_total, score_norm = out["total"], out["norm"]

            inst_id = inst.get("id", k)

            rows.append({
                "phase": phase,
                "gen": gen,
                "shard": shard_id,
                "ind_idx": idx,              # index in population for this generation
                "instance_id": inst_id,
                "score_total": float(score_total),
                "score_norm": float(score_norm),
            })

            total_sum += float(score_total)
            norm_sum += float(score_norm)

        # You can choose which fitness you optimize:
        # fitness = total_sum / n_inst          # fitness based on total
        fitness = norm_sum / n_inst         # OR based on normalized

        results.append((idx, fitness))

    pd.DataFrame.from_records(rows).to_parquet(out_path, index=False)
    return results

"""def _eval_individual(payload):
    idx, veh_policy, req_policy = payload

    if not _WORKER_INSTANCES:
        raise ValueError("Worker received empty instances list.")

    total = 0.0
    for inst in _WORKER_INSTANCES:
        total += _WORKER_SIM.run(
            inst,
            (veh_policy, req_policy),
            travel_metrics=_WORKER_travel_metrics
        )

    return idx, total / len(_WORKER_INSTANCES)"""

"""def evaluate_population_parallel(population, simulator, travel_metrics, instances, max_workers=None, chunksize=1):
    tasks = [(i, ind.veh_policy, ind.req_policy)
             for i, ind in enumerate(population)
             if ind.fitness is None]
    if not tasks:
        return

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(simulator, travel_metrics, instances),
    ) as ex:
        for idx, fitness in ex.map(_eval_individual, tasks, chunksize=chunksize):
            population[idx].fitness = fitness
            
def write_parquet_shard(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame.from_records(rows)
    df.to_parquet(out_path, index=False)  # uses pyarrow/fastparquet"""
    
def _chunked(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def evaluate_population_parallel(
    population,
    simulator,
    travel_metrics,
    instances,
    base_dir,
    phase,
    gen=None,
    max_workers=None,
    chunk_size=50,   # controls number of shard files
    force = False
):
    tasks = [(i, ind.veh_policy, ind.req_policy)
             for i, ind in enumerate(population)
             if force or ind.fitness is None]
    if not tasks:
        return

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(simulator, travel_metrics, instances),
    ) as ex:
        futures = []
        for shard_id, chunk in enumerate(_chunked(tasks, chunk_size)):
            payload = {
                "tasks": chunk,
                "base_dir": base_dir,
                "phase": phase,
                "gen": gen,
                "shard_id": shard_id,
            }
            futures.append(ex.submit(_eval_chunk_and_write, payload))

        # collect and assign fitness
        for fut in as_completed(futures):
            for idx, fitness in fut.result():
                population[idx].fitness = fitness