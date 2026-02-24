from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

PHASES = ("train", "validation", "test", "real", "cross_eval")

def _read_parquet(f):
    dfi = pd.read_parquet(f)
        #print(dfi["score_total"].max())
        #print(dfi[dfi["score_total"] > 1e8].head())
        # --- normalize common columns to avoid schema merge issues later ---
    if "phase" in dfi.columns:
        dfi["phase"] = dfi["phase"].astype("string")
    if "instance_id" in dfi.columns:
        dfi["instance_id"] = dfi["instance_id"].astype("string")
    if "ind_id" in dfi.columns:
        dfi["ind_id"] = dfi["ind_id"].astype("string")

    # gen can be missing (validation/test), keep nullable int
    if "gen" in dfi.columns:
        dfi["gen"] = pd.to_numeric(dfi["gen"], errors="coerce").astype("Int64")
        
    return dfi

def _read_parquet_files(files):
    dfs = []
    for f in files:
        dfi = _read_parquet(f=f)

        dfs.append(dfi)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def load_runs(base_dir: Path, return_by_phase: bool = True):
    runs_dir = base_dir / "runs"

    by_phase = {}
    for phase in PHASES:
        phase_dir = runs_dir / f"phase={phase}"
        if not phase_dir.exists():
            continue

        if phase == "train":
            # read each gen folder
            gen_dirs = sorted(phase_dir.glob("gen=*"))
            files = []
            for gd in gen_dirs:
                files.extend(sorted(gd.rglob("*.parquet")))
        else:
            files = sorted(phase_dir.rglob("*.parquet"))

        df_phase = _read_parquet_files(files)

        # If gen is encoded in the folder name but not in file columns,
        # we can parse it from the path (optional safety net).
        if phase == "train" and ("gen" not in df_phase.columns or df_phase["gen"].isna().all()):
            # parse gen=XXXX from each file path is not possible after concat;
            # better ensure you wrote gen into the rows. If you didn’t, tell me.
            pass

        by_phase[phase] = df_phase

    if return_by_phase:
        return by_phase

    # combined df
    if by_phase:
        return pd.concat(by_phase.values(), ignore_index=True)
    return pd.DataFrame()


"""def boxplot_scores(df_phase: pd.DataFrame, metric_col: str, title: str, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.boxplot(y=df_phase[metric_col])
    #sns.boxplot(x=df_phase["instance_id"], y=df_phase[metric_col])
    plt.title(title)
    plt.ylabel(metric_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()"""
    
def boxplot_scores(df_phase, metric_col, title, out_path, log_scale=True, clip = True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if clip:
        data = clipped_series(df_phase[metric_col], q=0.99)
    else:
        data = df_phase[metric_col].dropna()

    if data.empty:
        print(f"No data for {title}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=data, ax=ax)

    ax.set_title(title)
    ax.set_ylabel(metric_col)

    if log_scale:
        # log requires strictly positive values
        if (data <= 0).any():
            # fallback: symlog handles 0; small linear region near 0
            ax.set_yscale("symlog", linthresh=1.0)
        else:
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig) 



def plot_progress(summary_df: pd.DataFrame, title: str, out_path: Path):
    s = summary_df.sort_values("gen")
    x = s["gen"].astype(int)
    y = s["mean"]
    p25 = s["p25"]
    p75 = s["p75"]

    plt.figure(figsize=(10,4))
    plt.plot(x, y, label="mean")
    plt.fill_between(x, p25, p75, alpha=0.25, label="p25–p75")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def summarize(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series({"n": 0})
    return pd.Series({
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(ddof=1),
        "p25": s.quantile(0.25),
        "p75": s.quantile(0.75),
    })

def add_metric_summaries(df: pd.DataFrame, group_cols, metric_col: str) -> pd.DataFrame:
    out = df.groupby(group_cols)[metric_col].apply(summarize).unstack()
    out = out.reset_index()
    out["metric"] = metric_col
    return out

"""def load_runs(base_dir: Path) -> pd.DataFrame:
    runs_dir = base_dir / "runs"
    dfs = []
    for phase in ("test", "real", "validation", "train"):
        phasedir = Path(runs_dir/f"phase={phase}")
        if phasedir.exists():
            print(f"reading directory {phasedir}")
            if phase == "train":
                for 
            df = pd.read_parquet(phasedir)
            dfs.append(df)
        else:
            print (f"directory {phasedir} not found")
            continue
        # read all parquet files under runs/

    # Normalize dtypes / missing gens
    for df in dfs:
        if "gen" in df.columns:
            df["gen"] = pd.to_numeric(df["gen"], errors="coerce").astype("Int64")
    return dfs[1]"""

def per_individual_avg_over_instances(df_phase: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    # one value per individual: average over instances
    return (df_phase
            .groupby(["phase", "gen", "ind_idx"], dropna=False)[metric_col]
            .mean()
            .reset_index(name="avg_over_instances"))
    
def summarize_avg_fitness_across_instances(df_phase: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    per_ind = per_individual_avg_over_instances(df_phase, metric_col)
    return add_metric_summaries(per_ind, group_cols=["phase", "gen"], metric_col="avg_over_instances")

def per_individual_std_over_instances(df_phase: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    return (df_phase
            .groupby(["phase", "gen", "ind_idx"], dropna=False)[metric_col]
            .std(ddof=1)
            .reset_index(name="std_over_instances"))

def summarize_std_across_instances(df_phase: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    per_ind = per_individual_std_over_instances(df_phase, metric_col)
    return add_metric_summaries(per_ind, group_cols=["phase", "gen"], metric_col="std_over_instances")

def save_df_csv(df, out_path, index=False):
    """
    Save a DataFrame to CSV (creates parent dirs).
    Writes UTF-8, with ';' optional if you prefer Excel-friendly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index, encoding="utf-8")

def clipped_series(s, q=0.99):
    s = s.dropna()
    if s.empty:
        return s
    upper_cap = s.quantile(q)
    lower_cap = s.quantile((1-q)*5)
    return s.clip(upper=upper_cap, lower=lower_cap)

base_dir = Path("results/Höxter_v5_r200")   # example
summ_dir = base_dir / "summaries"
by_phase = load_runs(base_dir, return_by_phase=True)
train = by_phase.get("train", pd.DataFrame())
val   = by_phase.get("validation", pd.DataFrame())
test  = by_phase.get("test", pd.DataFrame())
boxplot_scores(test, "score_total", "Test: total score distribution over instances",
               base_dir / "summaries" / "test_total_box.png")
boxplot_scores(test, "score_norm", "Test: normalised score distribution over instances",
               base_dir / "summaries" / "test_norm_box.png")

norm_progress = summarize_avg_fitness_across_instances(train, "score_norm")
# norm_progress columns: phase, gen, n, mean, median, std, p25, p75, metric
plot_progress(norm_progress, "Training: normalised avg fitness over generations",
              base_dir / "summaries" / "train_norm_progress.png")


save_df_csv(train, summ_dir / "train_runs.csv")
save_df_csv(val,   summ_dir / "validation_runs.csv")
save_df_csv(test,  summ_dir / "test_runs.csv")


cross_eval_path = Path("/Users/markuswienkemeier/Coding/GPHH/results/cross_eval_2026-02-23_05-47-08/cross_eval.parquet")
cross_eval = _read_parquet(cross_eval_path)

save_df_csv(cross_eval, cross_eval_path.parent / "cross_eval.csv")