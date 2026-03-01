from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import math
import pandas as pd

@dataclass
class GenRuntimeResult:
    gen: int
    gen_dir: str
    n_shards: int
    n_obs: int
    t0: pd.Timestamp
    t_end_max: pd.Timestamp
    total_duration_s: float
    sec_per_obs: float

def _folder_start_time(gen_dir: Path) -> float:
    """
    Returns POSIX timestamp seconds for 'start time'.
    On macOS st_birthtime may exist; on other OS it may not.
    Fallback: st_mtime.
    """
    st = gen_dir.stat()
    birth = getattr(st, "st_birthtime", None)
    return float(birth if birth is not None else st.st_mtime)

def _parquet_num_rows(p: Path) -> int:
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(p).metadata.num_rows)

def estimate_generation_runtime_from_dir(
    gen_dir: str | Path,
    max_workers: int = 5,
    shard_glob: str = "shard=*.parquet",
) -> GenRuntimeResult:
    """
    Estimate runtime of a generation using:
      - t0 = folder start time (birthtime if available else mtime)
      - shard end times = shard file mtime
      - tasks scheduled on max_workers slots (greedy).

    Returns GenRuntimeResult incl. total_duration_s and sec_per_obs.
    """
    gen_dir = Path(gen_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(gen_dir)

    # parse gen id from folder name gen=XXXX if possible
    gen = None
    if gen_dir.name.startswith("gen="):
        try:
            gen = int(gen_dir.name.split("=", 1)[1])
        except Exception:
            gen = -1
    else:
        gen = -1

    shard_files = sorted(gen_dir.glob(shard_glob))
    if not shard_files:
        # fallback: any parquet in folder tree
        shard_files = sorted(gen_dir.rglob("*.parquet"))

    if not shard_files:
        return GenRuntimeResult(
            gen=gen, gen_dir=str(gen_dir), n_shards=0, n_obs=0,
            t0=pd.to_datetime(_folder_start_time(gen_dir), unit="s"),
            t_end_max=pd.to_datetime(_folder_start_time(gen_dir), unit="s"),
            total_duration_s=0.0, sec_per_obs=float("nan")
        )

    t0 = _folder_start_time(gen_dir)

    shard_info = []
    total_rows = 0

    for f in shard_files:
        end = float(f.stat().st_mtime)
        nrows = _parquet_num_rows(f)
        total_rows += nrows
        shard_info.append((end, nrows, str(f)))

    # sort by end time (earliest finished first)
    shard_info.sort(key=lambda x: x[0])

    # greedy slot scheduling:
    # each slot has "available_time"; a task starts at max(t0, earliest_available)
    slots = [t0] * max_workers
    total_duration = 0.0

    for end, nrows, fname in shard_info:
        # pick earliest available worker
        i = min(range(max_workers), key=lambda k: slots[k])
        start = max(t0, slots[i])

        # guard against weird timestamps
        if end < start:
            duration = 0.0
        else:
            duration = end - start

        total_duration += duration
        slots[i] = end  # worker is next available at task end

    sec_per_obs = (total_duration / total_rows) if total_rows > 0 else float("nan")

    t_end_max = max(end for end, _, _ in shard_info)

    return GenRuntimeResult(
        gen=gen,
        gen_dir=str(gen_dir),
        n_shards=len(shard_files),
        n_obs=int(total_rows),
        t0=pd.to_datetime(t0, unit="s"),
        t_end_max=pd.to_datetime(t_end_max, unit="s"),
        total_duration_s=float(total_duration),
        sec_per_obs=float(sec_per_obs),
    )

def estimate_all_generations_runtime(
    train_phase_dir: str | Path,
    out_csv: str | Path,
    max_workers: int = 5,
    shard_glob: str = "shard=*.parquet",
) -> pd.DataFrame:
    """
    Wrapper: iterate all gen=* folders under train_phase_dir and write one CSV.
    """
    train_phase_dir = Path(train_phase_dir)
    if not train_phase_dir.exists():
        raise FileNotFoundError(train_phase_dir)

    results = []
    for gen_dir in sorted(train_phase_dir.glob("gen=*")):
        if not gen_dir.is_dir():
            continue
        res = estimate_generation_runtime_from_dir(
            gen_dir, max_workers=max_workers, shard_glob=shard_glob
        )
        results.append(res.__dict__)

    df = pd.DataFrame(results).sort_values("gen")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df