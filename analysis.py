from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import re

#PHASES = ("train", "validation", "test", "real", "cross_eval")
class Analyser():
    def __init__(self):
        self.phases = ("train", "validation", "test", "real", "cross_eval")
        self.cities = ("Paderborn", "Höxter")
        self.req_params = ("20", "50", "200")
        self.veh_params = ("2", "5", "20")
        self.PROJECT_ROOT = Path(__file__).resolve().parent
        self.results_dir = self.PROJECT_ROOT / "results"
        self.summary_dir = self.results_dir / "summary"
        self.runsets = [p for p in self.results_dir.iterdir() if p.is_dir() and not p.name.startswith("cross_eval") and not p.name == "summary"]
        self.rel_runs = Path("runs")
        self.rel_train = self.rel_runs / "phase=train"
        self.rel_validaton = self.rel_runs / "phase=validation"
        self.rel_test = self.rel_runs / "phase=test"
        self.rel_real = self.rel_runs / "phase=real"
        self.joined_train_files_paths = self.ensure_joined_train_files()
        
    def ensure_joined_train_files(self):
        files = []
        for run in self.runsets:
            train_dir = run / self.rel_train 
            file = train_dir / "train_joined.parquet"
            if not file.exists():
                self.join_train_files(train_dir=train_dir)
            files.append(file)
        return files
        
    def join_train_files(
        self,
        train_dir,
        write_parquet: bool = True,
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Join all train shard parquet files across all generations into one DataFrame.

        Expected directory layout:
            train_dir/
              gen=0000/
                shard=000.parquet
                shard=001.parquet
              gen=0001/
                shard=...

        - Reads all *.parquet under all gen=* folders
        - Injects/normalizes 'gen' from folder name if missing
        - Normalizes dtypes to avoid pyarrow schema merge issues

        Parameters
        ----------
        train_dir : str | Path
            Path to runs/phase=train directory.
        out_path : str | Path | None
            If given, write the joined dataframe to this file.
            Suffix decides format if write_parquet/write_csv are both False.
        write_parquet : bool
            If True and out_path is None, writes train_joined.parquet next to train_dir.
        write_csv : bool
            If True and out_path is None, writes train_joined.csv next to train_dir.

        Returns
        -------
        pd.DataFrame
            Concatenated train runs across all gens and shards.
        """
        train_dir = Path(train_dir)
        if not train_dir.exists():
            raise FileNotFoundError(f"train_dir not found: {train_dir}")

        gen_dirs = sorted(train_dir.glob("gen=*"))
        if not gen_dirs:
            # fallback: maybe files are directly under train_dir
            files = sorted(train_dir.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found under: {train_dir}")
            df = self._read_and_normalize_parquet_files(files)
            return df

        dfs: list[pd.DataFrame] = []
        gen_re = re.compile(r"gen=(\d+)")

        for gd in gen_dirs:
            m = gen_re.search(gd.name)
            gen = int(m.group(1)) if m else None

            files = sorted(gd.rglob("*.parquet"))
            if not files:
                continue
            
            dfg = self._read_and_normalize_parquet_files(files)

            # Ensure gen column exists and is correct
            if "gen" not in dfg.columns:
                dfg["gen"] = gen
            else:
                # fill missing with folder gen; keep file gen if present
                dfg["gen"] = pd.to_numeric(dfg["gen"], errors="coerce").astype("Int64")
                if gen is not None:
                    dfg.loc[dfg["gen"].isna(), "gen"] = gen

            # Ensure phase is present (some shards may omit it)
            if "phase" not in dfg.columns:
                dfg["phase"] = "train"

            dfs.append(dfg)

        if not dfs:
            raise FileNotFoundError(f"No parquet files found under any gen=* in {train_dir}")

        df = pd.concat(dfs, ignore_index=True)

        # Final normalization (important for stability)
        if "phase" in df.columns:
            df["phase"] = df["phase"].astype("string")
        if "instance_id" in df.columns:
            df["instance_id"] = df["instance_id"].astype("string")
        if "gen" in df.columns:
            df["gen"] = pd.to_numeric(df["gen"], errors="coerce").astype("Int64")

        # Optional: write output
        if write_csv:
            df.to_csv(Path(train_dir / "train_joined.csv"), index=False, encoding="utf-8")
        if write_parquet:
            # default parquet
            df.to_parquet(Path(train_dir / "train_joined.parquet"), index=False)
        return df
    
    @staticmethod
    def _read_and_normalize_parquet_files(files: list[Path]) -> pd.DataFrame:
        """
        Read parquet files one-by-one to avoid pyarrow dataset schema merge errors.
        Normalizes common columns to stable dtypes.
        """
        dfs: list[pd.DataFrame] = []
        for f in files:
            dfi = pd.read_parquet(f)

            # Normalize common problematic columns
            if "phase" in dfi.columns:
                dfi["phase"] = dfi["phase"].astype("string")
            if "instance_id" in dfi.columns:
                dfi["instance_id"] = dfi["instance_id"].astype("string")
            if "ind_id" in dfi.columns:
                dfi["ind_id"] = dfi["ind_id"].astype("string")

            if "gen" in dfi.columns:
                dfi["gen"] = pd.to_numeric(dfi["gen"], errors="coerce").astype("Int64")

            dfs.append(dfi)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def clip_with_count(self, df: pd.DataFrame, col: str, upper: float = None, lower: float = None):
        s = pd.to_numeric(df[col], errors="coerce")
        if upper:
            # count how many would be clipped (exclude NaNs)
            n_clipped = (s > upper).sum()

            # apply clipping
            df = df.copy()
            df[col] = s.clip(upper=upper)
        elif lower:
            # count how many would be clipped (exclude NaNs)
            n_clipped = (s < lower).sum()

            # apply clipping
            df = df.copy()
            df[col] = s.clip(lower=lower)
        else:
            print("No bound to clip given")
        return df, int(n_clipped)
    
    def plot_phase_norm_mean_std_per_gen(self, df: pd.DataFrame, out_path: str | Path, run_prefix: str = None, phase: str = "train", upper_bound: int = 15, show_n = False):
        df_data = df[df["phase"] == phase].copy()
        df_data["gen"] = pd.to_numeric(df_data["gen"], errors="coerce")
        df_data, n_clipped = self.clip_with_count(df = df_data, col="score_norm", upper= upper_bound)
        print(n_clipped)
        df_data = df_data.dropna(subset=["gen", "score_norm"])
        df_data["gen"] = df_data["gen"].astype(int)

        # per-generation summary
        summ = (df_data.groupby("gen")["score_norm"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
                .sort_values("gen"))

        summ["lower"] = summ["mean"] - summ["std"]
        summ["upper"] = summ["mean"] + summ["std"]

        # log scale requires positive values
        eps = 1e-12
        summ["lower"] = summ["lower"].clip(lower=eps)
        summ["mean"]  = summ["mean"].clip(lower=eps)
        summ["upper"] = summ["upper"].clip(lower=eps)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4))

        # score on left axis (log)
        ax.plot(summ["gen"], summ["mean"], color="tab:blue", label="mean(score_norm)")
        ax.fill_between(summ["gen"], summ["lower"], summ["upper"],
                        color="tab:blue", alpha=0.25, label="±1 std")
#        ax.set_yscale("log")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Normalised score ")
        lines, labels = ax.get_legend_handles_labels()
        
        if show_n:
            # counts on right axis (linear)
            ax2 = ax.twinx()
            ax2.plot(summ["gen"], summ["n"], color="tab:orange", linewidth=2, label="n observations")
            ax2.set_ylabel("Observations (count)")

            # combined legend

            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax.legend(lines, labels, loc="best")

        fig.tight_layout()
        if out_path.is_dir():
            png_default_name = "mean and error per generation.png"
            csv_default_name = "mean and error per generation.csv"
            if run_prefix:
                png_default_name = run_prefix + png_default_name
                csv_default_name = run_prefix + csv_default_name
            csv_path = out_path / csv_default_name
            out_path = out_path / png_default_name
            summ.to_csv(csv_path)
        fig.savefig(out_path)
        plt.close(fig)
    
    def plot_all_train(self):
        for filepath in self.joined_train_files_paths:
            run_prefix = filepath.resolve().parent.parent.parent.name 
            train_data = self._read_parquet(filepath)
            out_dir = self.summary_dir / "train"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.plot_phase_norm_mean_std_per_gen(df=train_data, out_path=out_dir, run_prefix=run_prefix, phase="train", upper_bound=15, show_n=True)
            print(run_prefix)
            
    def plot_all_validation(self):
        for run in self.runsets:
            run_prefix = run.name
            val_dir = run / self.rel_validaton
            val_data = self._read_and_normalize_parquet_files(val_dir.iterdir())
            val_data = val_data.drop(columns=["gen"])
            val_data = val_data.rename(columns={"ind_idx": "gen"})
            # make it a proper integer column (nullable)
            val_data["gen"] = pd.to_numeric(val_data["gen"], errors="coerce").astype("Int64")
            out_dir = self.summary_dir / "validation"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.plot_phase_norm_mean_std_per_gen(df=val_data, out_path=out_dir, run_prefix=run_prefix, phase="validation", upper_bound=500000)
    
    def _read_parquet(self, f) -> pd.DataFrame:
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

    def _read_parquet_files(self, files):
        dfs = []
        for f in files:
            dfi = self._read_parquet(f=f)

            dfs.append(dfi)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_runs(self, base_dir: Path, return_by_phase: bool = True):
        runs_dir = base_dir / "runs"

        by_phase = {}
        for phase in self.phases:
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

            df_phase = self._read_parquet_files(files)

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
        lower_cap = s.quantile((1-q))
        return s.clip(upper=upper_cap, lower=lower_cap)

    def analyse_cross_eval(
        self,
        df: pd.DataFrame = None,
        out_dir: str | Path = None,
        score_col: str = "score_norm",
        policy_cols = ("policy_city", "policy_r", "policy_v"),
        testset_cols = ("test_city", "test_r", "test_v"),
        clip_upper: float | None = None,   # e.g. 15.0 if you have error spikes
    ):
        """
        Cross-eval analysis:
        1) best policy: lowest average norm score over all test sets (and instances)
        2) most robust: lowest std dev over all test sets (and instances)
        3) most flexible: lowest worst testset mean (minimize maximum mean across test sets)
        4) stacked bars: mean and std split by test set for each policy

        Returns:
          policy_summary_df, per_policy_testset_df, winners_dict
        """
        if not out_dir:
            out_dir = self.summary_dir / "cross_val"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not df:
            first_cross_eval_dir = next(
                (p for p in self.results_dir.iterdir() if p.is_dir() and p.name.startswith("cross_eval")),
                None,  # default if nothing found
            )
            cross_eval_file = first_cross_eval_dir / "cross_eval.parquet"
            print(cross_eval_file)
            df = self._read_parquet(cross_eval_file)

        d = df.copy()

        # Filter to cross_eval only if present
        if "phase" in d.columns:
            d = d[d["phase"].astype(str) == "cross_eval"].copy()

        # Ensure score numeric
        d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
        d = d.dropna(subset=[score_col])

        # Optional clipping of crazy outliers (upper bound)
        if clip_upper is not None:
            d[score_col] = d[score_col].clip(upper=clip_upper)

        # Build readable IDs
        d["policy_id"] = d[list(policy_cols)].astype(str).agg(" | ".join, axis=1)
        d["testset_id"] = d[list(testset_cols)].astype(str).agg(" | ".join, axis=1)

        # ---- Per-policy-per-testset aggregation (this is your item 4 base table) ----
        per_policy_testset = (
            d.groupby(["policy_id", "testset_id"])[score_col]
             .agg(mean="mean", std="std", n="count")
             .reset_index()
        )

        # Save this table for inspection
        per_policy_testset.to_csv(out_dir / "per_policy_per_testset.csv", index=False, encoding="utf-8")

        # ---- Policy-level metrics (items 1-3) ----
        # Average & std across ALL runs across all testsets (run-level pooling)
        policy_overall = (
            d.groupby("policy_id")[score_col]
             .agg(avg_norm_all="mean", std_norm_all="std", n_runs="count")
             .reset_index()
        )

        # "Worst test set mean": for each policy, compute mean per testset then take max
        worst_testset = (
            per_policy_testset.groupby("policy_id")["mean"]
            .max()
            .reset_index(name="worst_testset_mean_norm")
        )

        policy_summary = policy_overall.merge(worst_testset, on="policy_id", how="left")

        # Winners for (1)-(3)
        best_policy_id = policy_summary.loc[policy_summary["avg_norm_all"].idxmin(), "policy_id"]
        most_robust_id = policy_summary.loc[policy_summary["std_norm_all"].idxmin(), "policy_id"]
        most_flexible_id = policy_summary.loc[policy_summary["worst_testset_mean_norm"].idxmin(), "policy_id"]

        winners = {
            "best_policy_id": best_policy_id,
            "most_robust_policy_id": most_robust_id,
            "most_flexible_policy_id": most_flexible_id,
        }

        # Save policy summary
        policy_summary.sort_values("avg_norm_all").to_csv(out_dir / "policy_summary.csv", index=False, encoding="utf-8")

        # Also write a tiny winners table
        pd.DataFrame([winners]).to_csv(out_dir / "winners.csv", index=False, encoding="utf-8")

        # ---- Stacked bar charts (means and stds split by test set) ----
        # Pivot so rows=policy, cols=testset
        mean_pivot = per_policy_testset.pivot(index="policy_id", columns="testset_id", values="mean").fillna(0.0)
        std_pivot  = per_policy_testset.pivot(index="policy_id", columns="testset_id", values="std").fillna(0.0)

        # Sort policies by overall avg to make plot readable
        order = (policy_summary.sort_values("avg_norm_all")["policy_id"]).tolist()
        mean_pivot = mean_pivot.reindex(order)
        std_pivot  = std_pivot.reindex(order)

        def _stacked_bar(pivot_df: pd.DataFrame, title: str, ylabel: str, out_path: Path):
            ax = pivot_df.plot(kind="bar", stacked=True, figsize=(14, 6), width=0.9)
            ax.set_title(title)
            ax.set_xlabel("Policy")
            ax.set_ylabel(ylabel)
            ax.legend(title="Test set", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            plt.close()

        _stacked_bar(
            mean_pivot,
            title="Cross-eval: mean normalised score per policy (stacked by test set)",
            ylabel="Mean score_norm (lower is better)",
            out_path=out_dir / "stacked_means_by_testset.png",
        )

        _stacked_bar(
            std_pivot,
            title="Cross-eval: std dev of normalised score per policy (stacked by test set)",
            ylabel="Std(score_norm)",
            out_path=out_dir / "stacked_stds_by_testset.png",
        )

        return policy_summary, per_policy_testset, winners

"""
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
"""

#
analyser = Analyser()
#analyser.join_train_files(train_dir="/Users/markuswienkemeier/Coding/GPHH/results/Höxter_v2_r20/runs/phase=train", write_parquet=True, write_csv=True)
analyser.ensure_joined_train_files()
#train_file = analyser.joined_train_files_paths[1]
#train_data = analyser._read_parquet(train_file)
#analyser.plot_train_norm_mean_std_ln(df = train_data, out_path = Path(train_file).resolve().parent)
#analyser.plot_all_validation()
#analyser.plot_all_train()
#policy_summary, per_policy_testset, winners = analyser.analyse_cross_eval()
#print(winners)
