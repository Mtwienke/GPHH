import os
import re
import numpy as np
from collections import OrderedDict
import numpy as np
from math import inf
import heapq
from pathlib import Path
from tqdm import tqdm

BIG = 1e9

class SourceFileStoreNPY:
    def __init__(self, graph, cache_dir, time_attr="travel_time", dist_attr="length"):
        self.G = graph
        self.cache_dir = cache_dir
        self.time_attr = time_attr
        self.dist_attr = dist_attr
        os.makedirs(cache_dir, exist_ok=True)

    def paths(self, u):
        uid = _safe_id(u)
        base = os.path.join(self.cache_dir, f"src_{uid}")
        return base + "_v.npy", base + "_t.npy", base + "_d.npy"

    def exists(self, u):
        pv, pt, pd = self.paths(u)
        return os.path.exists(pv) and os.path.exists(pt) and os.path.exists(pd)

    def _atomic_save_npy(self, path, arr):
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)

    def build(self, u):
        """
        Compute Dijkstra once for source u and store arrays on disk.
        Requires: dijkstra_time_and_dist(...) -> dict v -> (t, d)
        """
        lengths = dijkstra_time_and_dist(
            self.G, u, time_attr=self.time_attr, dist_attr=self.dist_attr
        )

        # Convert to arrays
        vs = np.fromiter(lengths.keys(), dtype=np.int64)
        td = np.array([lengths[v] for v in lengths], dtype=np.float32)  # (k,2)
        t = td[:, 0]
        d = td[:, 1]

        # sort by target node id for fast binary-search lookup
        order = np.argsort(vs)
        vs, t, d = vs[order], t[order], d[order]

        pv, pt, pd = self.paths(u)
        self._atomic_save_npy(pv, vs)
        self._atomic_save_npy(pt, t)
        self._atomic_save_npy(pd, d)

    def ensure(self, u):
        if not self.exists(u):
            self.build(u)

    def load(self, u, mmap=False):
        """
        Load arrays for source u.
        mmap=True uses memory mapping (low RAM, good if files are large).
        """
        self.ensure(u)
        pv, pt, pd = self.paths(u)
        mmode = "r" if mmap else None
        v = np.load(pv, mmap_mode=mmode)
        t = np.load(pt, mmap_mode=mmode)
        d = np.load(pd, mmap_mode=mmode)
        return v, t, d
    
class TravelMetricsDiskBackedLRU:
    def __init__(self, store: SourceFileStoreNPY, max_sources_in_ram=50, mmap_files=False):
        self.store = store
        self.max_sources_in_ram = max_sources_in_ram
        self.mmap_files = mmap_files
        self._ram = OrderedDict()  # u -> (v_arr, t_arr, d_arr)

    def _get_source(self, u):
        if u in self._ram:
            self._ram.move_to_end(u)
            return self._ram[u]

        v, t, d = self.store.load(u, mmap=self.mmap_files)
        self._ram[u] = (v, t, d)
        self._ram.move_to_end(u)

        if len(self._ram) > self.max_sources_in_ram:
            self._ram.popitem(last=False)

        return v, t, d

    def time_dist(self, u, v):
        v_arr, t_arr, d_arr = self._get_source(u)
        v = np.int64(v)

        idx = np.searchsorted(v_arr, v)
        if idx < len(v_arr) and v_arr[idx] == v:
            return float(t_arr[idx]), float(d_arr[idx])
        return BIG, BIG

    def get_time(self, u, v):
        return self.time_dist(u, v)[0]

    def get_dist(self, u, v):
        return self.time_dist(u, v)[1]

def _safe_id(x):
    s = str(x)
    return re.sub(r"[^0-9A-Za-z_.-]", "_", s)


def dijkstra_time_and_dist(G, source, time_attr="travel_time", dist_attr="length"):
    """
    Dijkstra that minimizes travel_time, and tracks distance along the chosen
    (time-optimal) path. For ties in time, it picks smaller distance.
    Returns: dict node -> (time, dist)
    """
    best = {source: (0.0, 0.0)}  # node -> (time, dist)
    pq = [(0.0, 0.0, source)]    # (time, dist, node)

    while pq:
        t, d, u = heapq.heappop(pq)
        if (t, d) != best.get(u, (inf, inf)):
            continue

        # iterate neighbors
        for v, keydict in G.adj[u].items():  # MultiDiGraph: v -> {key: attrdict}
            # choose among parallel edges by comparing (time, dist)
            # (we need to consider all keys because the time-optimal edge might differ)
            for k, attr in keydict.items():
                et = float(attr.get(time_attr, inf))
                ed = float(attr.get(dist_attr, 0.0))
                nt, nd = t + et, d + ed

                cur = best.get(v, (inf, inf))
                # Lexicographic: smaller time, then smaller distance
                if (nt < cur[0]) or (nt == cur[0] and nd < cur[1]):
                    best[v] = (nt, nd)
                    heapq.heappush(pq, (nt, nd, v))

    return best

class travel_metrics():
    def __init__(self, graph, city, cache_dir_name = "travelmetrics", ram_storage=50, mmap_files=False):
        self.graph =graph
        PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
        CACHE_ROOT = PROJECT_ROOT / cache_dir_name
        cachepath = CACHE_ROOT / city
        self.store = SourceFileStoreNPY(graph=graph, cache_dir=cachepath)
        self.load = TravelMetricsDiskBackedLRU(self.store, ram_storage, mmap_files=mmap_files)
        
    def get_time(self, u, v):
        return self.load.get_time(u, v)
    
    def get_distance(self, u, v):
        return self.load.get_dist(u, v)
    
    def build_cache(self):
        sources = tqdm(self.graph.nodes, "Generating cache files, 1 to all")
        for u in sources:
            self.store.ensure(u)