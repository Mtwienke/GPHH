from dataclasses import dataclass
from enum import Enum, auto
import csv
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def sec_to_hhmmss(x, pos=None):
    x = int(max(0, x))
    h = x // 3600
    m = (x % 3600) // 60
    s = x % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


REQ_COLORS = {
    "UNKNOWN": "lightgrey",
    "LOST": "red",
    "WAITING": "orange",
    "ASSIGNED": "gold",
    "IN_SERVICE": "seagreen",
    "COMPLETED": "dodgerblue",
}

def plot_request_segments_png(request_segments_csv, out_file=None, out_dir=None,
                              max_requests=80, sort_by="request_id",
                              figsize=(14, 10), dpi=200):
    """
    Plot request segments as a Gantt-like chart and save as PNG.

    Parameters
    ----------
    request_segments_csv : str | Path
        Path to request_segments.csv with columns:
        request_id, t_start, t_end, state, vehicle_id (vehicle_id optional)
    out_file : str | Path | None
        Full output file path (e.g. ".../requests.png"). If given, used directly.
    out_dir : str | Path | None
        Output directory (used if out_file is None). File will be "requests.png".
    max_requests : int
        Limit number of requests plotted (for readability).
    sort_by : str
        "request_id" or "t_start" (sort requests by earliest segment start).
    """
    request_segments_csv = Path(request_segments_csv)
    if not request_segments_csv.exists():
        return
    df = pd.read_csv(request_segments_csv)
    if df.empty:
        return None

    # Basic cleanup
    df["t_start"] = df["t_start"].astype(float)
    df["t_end"] = df["t_end"].astype(float)
    df = df[df["t_end"] > df["t_start"]].copy()

    # Choose which requests to show
    if sort_by == "t_start":
        order = (df.groupby("request_id")["t_start"].min()
                   .sort_values()
                   .index.tolist())
    else:
        order = sorted(df["request_id"].unique().tolist())

    order = order[:max_requests]
    df = df[df["request_id"].isin(order)].copy()

    # Map request_id -> y position
    y_map = {rid: i for i, rid in enumerate(order)}
    df["y"] = df["request_id"].map(y_map)

    # Output path
    if out_file is None:
        out_dir = Path(out_dir) if out_dir is not None else request_segments_csv.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "requests.png"
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    bar_h = 0.8
    for _, row in df.iterrows():
        color = REQ_COLORS.get(row["state"], "black")
        ax.broken_barh(
            [(row["t_start"], row["t_end"] - row["t_start"])],
            (row["y"] - bar_h/2, bar_h),
            facecolors=color,
            edgecolors="white",
            linewidth=0.8
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmmss))
    ax.set_xlabel("time (hh:mm:ss)")
    ax.set_ylabel("request_id")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("Request timelines (segments)")

    # Legend
    handles = [plt.Line2D([0], [0], color=c, lw=8) for c in REQ_COLORS.values()]
    labels = list(REQ_COLORS.keys())
    ax.legend(handles, labels, loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file


def export_dataclasses_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

class ReqState(Enum):
    UNKNOWN = auto()
    WAITING = auto()
    ASSIGNED = auto()
    IN_SERVICE = auto()
    COMPLETED = auto()
    LOST = auto()

@dataclass
class RequestSegment:
    request_id: int
    t_start: float
    t_end: float
    state: str
    vehicle_id: int | None = None

class RequestRecorder:
    def __init__(self, ignore_types=None, ignore_ids=None):
        self.segments: list[RequestSegment] = []
        self._open: dict[int, tuple[float, ReqState, int | None]] = {}
        self.ignore_types = set(ignore_types or [])
        self.ignore_ids = set(ignore_ids or [])

    def _ignore(self, req) -> bool:
        rid = getattr(req, "id", None)
        rtype = getattr(req, "request_type", None)
        return (rid in self.ignore_ids) or (rtype in self.ignore_types)

    def start_waiting(self, req, t: float):
        """Call at request arrival (for real ride requests)."""
        if self._ignore(req):
            return
        rid = req.id
        # start/open segment at WAITING
        self._open[rid] = (t, ReqState.WAITING, None)

    def switch(self, req, t: float, new_state: ReqState, vehicle_id: int | None = None):
        """Close current segment and open a new one."""
        if self._ignore(req):
            return
        rid = req.id

        # if nothing open yet, treat as starting now (no UNKNOWN segment stored)
        if rid not in self._open:
            self._open[rid] = (t, new_state, vehicle_id)
            return

        t0, old_state, old_vid = self._open[rid]
        if t > t0:
            self.segments.append(RequestSegment(
                request_id=rid, t_start=t0, t_end=t,
                state=old_state.name, vehicle_id=old_vid
            ))
        self._open[rid] = (t, new_state, vehicle_id)

    def close_all(self, t_end: float):
        """Close remaining open segments at end of simulation."""
        for rid, (t0, st, vid) in list(self._open.items()):
            if t_end > t0:
                self.segments.append(RequestSegment(
                    request_id=rid, t_start=t0, t_end=t_end,
                    state=st.name, vehicle_id=vid
                ))
        self._open.clear()