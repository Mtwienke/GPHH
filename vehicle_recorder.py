from dataclasses import dataclass, asdict
from enum import Enum, auto
import csv
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

def sec_to_hhmmss(x, pos=None):
    x = int(max(0, x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

class VehState(Enum):
    IDLE = auto()
    DRIVE_TO_PICKUP = auto()
    FULFILL_REQUEST = auto()
    CHARGING = auto()

@dataclass
class StateSegment:
    vehicle_id: int
    t_start: float
    t_end: float
    state: str
    request_id: int | None = None
    station_id: int | None = None

@dataclass
class SoCPoint:
    vehicle_id: int
    time: float
    soc: float

class ActivityRecorder:
    def __init__(self):
        self.segments: list[StateSegment] = []
        self.soc: list[SoCPoint] = []
        # offene Segmente pro Vehicle
        self._open: dict[int, tuple[float, VehState, int | None, int | None]] = {}

    def start(self, vehicle_id: int, t: float, state: VehState, soc: float,
              request_id=None, station_id=None):
        self._open[vehicle_id] = (t, state, request_id, station_id)
        self.soc.append(SoCPoint(vehicle_id, t, soc))

    def switch(self, vehicle_id: int, t: float, new_state: VehState, soc: float,
               request_id=None, station_id=None):
        t0, old_state, old_rid, old_sid = self._open[vehicle_id]
        if t > t0:
            self.segments.append(StateSegment(
                vehicle_id=vehicle_id, t_start=t0, t_end=t,
                state=old_state.name, request_id=old_rid, station_id=old_sid
            ))
        self._open[vehicle_id] = (t, new_state, request_id, station_id)
        self.soc.append(SoCPoint(vehicle_id, t, soc))

    def close_all(self, t_end: float, fleet: dict[int, "Vehicle"]):
        for vid, (t0, st, rid, sid) in list(self._open.items()):
            if t_end > t0:
                self.segments.append(StateSegment(
                    vehicle_id=vid, t_start=t0, t_end=t_end,
                    state=st.name, request_id=rid, station_id=sid
                ))
            self.soc.append(SoCPoint(vid, t_end, fleet[vid].charge_level / fleet[vid].battery_capacity))
        self._open.clear()

def export_csv(rec: ActivityRecorder, segments_path: str, soc_path: str):
    with open(segments_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(rec.segments[0]).keys())
        w.writeheader()
        for s in rec.segments:
            w.writerow(asdict(s))

    with open(soc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(rec.soc[0]).keys())
        w.writeheader()
        for p in rec.soc:
            w.writerow(asdict(p))

STATE_COLORS = {
    "IDLE": "lightgrey",
    "CHARGING": "gold",
    "DRIVE_TO_PICKUP": "dodgerblue",
    "FULFILL_REQUEST": "seagreen",
}

def plot_vehicle_png(segments_csv, soc_csv, out_dir):
    seg = pd.read_csv(segments_csv)
    soc = pd.read_csv(soc_csv)

    for vid in sorted(seg["vehicle_id"].unique()):
        seg_v = seg[seg["vehicle_id"] == vid]
        soc_v = soc[soc["vehicle_id"] == vid].sort_values("time")

        fig, ax = plt.subplots(figsize=(14, 3))
        for _, row in seg_v.iterrows():
            ax.broken_barh(
                [(row["t_start"], row["t_end"] - row["t_start"])],
                (0, 10),
                facecolors=STATE_COLORS.get(row["state"], "black"),
                edgecolors="white",
                linewidth=1,
            )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmmss))
        ax.set_xlabel("time (hh:mm:ss)")
        ax.set_yticks([])
        ax.set_xlabel("time")
        ax.set_title(f"Vehicle {vid}: activity + SoC")

        ax2 = ax.twinx()
        ax2.plot(soc_v["time"], soc_v["soc"], color="black", linewidth=1.5)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("SoC")

        fig.tight_layout()
        fig.savefig(f"{out_dir}/vehicle_{vid}.png", dpi=200)
        plt.close(fig)