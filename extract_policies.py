from __future__ import annotations
import re
import csv
from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Globale Liste der erlaubten Terminals (nur Namen, nicht die Funktionsobjekte)
TERMINAL_NAMES = [
    "ISCR", "ISIR", "ISWV", "TVPU", "COST", "DEM", "DUR", "SLACK", "RT", "TVC",
    "AR", "OBV", "LCT", "CRDV", "CRPP", "CRPD", "CRDP", "CRDD",
]

TERM_RE = re.compile(r"TerminalNode\(name='([^']+)'\)")

def _count_terminals(policy_str: str) -> dict[str, int]:
    if not isinstance(policy_str, str):
        return {}
    names = TERM_RE.findall(policy_str)
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return counts

def make_terminal_boxplots_request_vs_vehicle(
    best_policies_csv: str | Path,
    out_dir: str | Path,
):
    best_policies_csv = Path(best_policies_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(best_policies_csv)

    rows = []
    for _, r in df.iterrows():
        city = r["city"]
        n_requests = int(r["n_requests"])
        n_vehicles = int(r["n_vehicles"])

        req_counts = _count_terminals(r["Request_Policy"])
        veh_counts = _count_terminals(r["Vehicle_Policy"])

        # Für jedes Szenario: ALLE erlaubten Terminals eintragen (fehlende -> 0)
        for t in TERMINAL_NAMES:
            rows.append({
                "city": city,
                "n_requests": n_requests,
                "n_vehicles": n_vehicles,
                "terminal": t,
                "count": req_counts.get(t, 0),
                "part": "request",
            })
            rows.append({
                "city": city,
                "n_requests": n_requests,
                "n_vehicles": n_vehicles,
                "terminal": t,
                "count": veh_counts.get(t, 0),
                "part": "vehicle",
            })

    long = pd.DataFrame(rows)

    # Terminal-Reihenfolge fixieren (damit alle Plots gleich sortiert sind)
    long["terminal"] = pd.Categorical(long["terminal"], categories=TERMINAL_NAMES, ordered=True)

    long.to_csv(out_dir / "terminal_counts_long_request_vehicle.csv", index=False, encoding="utf-8")

    def _boxplot_rv(data: pd.DataFrame, title: str, path: Path):
        if data.empty:
            print(f"Skip (no data): {title}")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(max(10, 0.5 * len(TERMINAL_NAMES)), 5))
        ax = sns.boxplot(
            data=data,
            x="terminal",
            y="count",
            hue="part",
            dodge=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Terminal")
        ax.set_ylabel("Anzahl Vorkommen")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Teilpolicy", loc="upper right")

        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # 1) Alle Szenarien
    _boxplot_rv(
        long,
        "Terminal-Vorkommen – alle Szenarien (Request vs Vehicle)",
        out_dir / "01_all_scenarios_request_vs_vehicle.pdf",
    )

    # 2) Nach n_requests (je Wert ein Plot)
    for nr in sorted(long["n_requests"].unique()):
        sub = long[long["n_requests"] == nr]
        _boxplot_rv(
            sub,
            f"Terminal-Vorkommen – n_requests={nr} (Request vs Vehicle)",
            out_dir / f"02_n_requests_{nr}_request_vs_vehicle.pdf",
        )

    # 3) Nach n_vehicles (je Wert ein Plot)
    for nv in sorted(long["n_vehicles"].unique()):
        sub = long[long["n_vehicles"] == nv]
        _boxplot_rv(
            sub,
            f"Terminal-Vorkommen – n_vehicles={nv} (Request vs Vehicle)",
            out_dir / f"03_n_vehicles_{nv}_request_vs_vehicle.pdf",
        )


def extract_policies_to_csv(txt_path, csv_path):
    """
    Extracts runs from a log-like .txt and writes a CSV with:
    city, n_requests, n_vehicles, Request_Policy, Vehicle_Policy

    Assumes each run starts with the 'starting run for ...' line and later
    contains (in any order) the two policy lines.
    """
    txt_path = Path(txt_path)
    csv_path = Path(csv_path)

    rows = []
    current = None

    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = START_RE.match(line)
            if m:
                # finalize previous run if present (even if incomplete)
                if current is not None:
                    rows.append(current)

                current = {
                    "city": m.group("city"),
                    "n_requests": int(m.group("n_requests")),
                    "n_vehicles": int(m.group("n_vehicles")),
                    "Request_Policy": "",
                    "Vehicle_Policy": "",
                }
                continue

            if current is None:
                continue

            m = REQ_RE.match(line)
            if m:
                current["Request_Policy"] = m.group("policy")
                continue

            m = VEH_RE.match(line)
            if m:
                current["Vehicle_Policy"] = m.group("policy")
                continue

    if current is not None:
        rows.append(current)

    # Optionally: keep only runs where both policies were found
    rows = [r for r in rows if r["Request_Policy"] and r["Vehicle_Policy"]]

    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=["city", "n_requests", "n_vehicles", "Request_Policy", "Vehicle_Policy"],
        )
        writer.writeheader()
        writer.writerows(rows)


# Example usage:
#extract_policies_to_csv(r"G:\Desktop\GPHH\Programm\GPHH\results_1\Terminal Saved Output.txt", r"G:\Desktop\GPHH\Programm\GPHH\results_1\Terminal Saved Output.csv")
make_terminal_boxplots_request_vs_vehicle(r"G:\Desktop\GPHH\Programm\GPHH\results_1\best_policies.csv", r"G:\Desktop\GPHH\Programm\GPHH\results_1\summary")