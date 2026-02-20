from math import inf
import heapq
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import json
from osmnx import graph
from pathlib import Path
import os
import re
import numpy as np
      
# Globals inside each worker process
_G = None
_TIME_ATTR = None
_DIST_ATTR = None

def init_worker(graph, time_attr, dist_attr):
    global _G, _TIME_ATTR, _DIST_ATTR
    _G = graph
    _TIME_ATTR = time_attr
    _DIST_ATTR = dist_attr

def compute_row(args):
    """
    args: (row_index, source_node, node_to_idx, key_nodes, BIG_T, BIG_D)
    returns: (row_index, tt_row, td_row)
    """
    row_i, u, node_to_idx, key_nodes, BIG_T, BIG_D = args

    # your custom Dijkstra returning {v: (t,d)}
    lengths = dijkstra_time_and_dist(_G, u, time_attr=_TIME_ATTR, dist_attr=_DIST_ATTR)

    n = len(key_nodes)
    tt_row = np.full(n, BIG_T, dtype=np.float32)
    td_row = np.full(n, BIG_D, dtype=np.float32)

    # Fill reachable targets that are in key_nodes
    for v, (t, d) in lengths.items():
        j = node_to_idx.get(v)
        if j is not None:
            tt_row[j] = np.float32(t)
            td_row[j] = np.float32(d)

    tt_row[row_i] = 0.0
    td_row[row_i] = 0.0
    return row_i, tt_row, td_row

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

def build_dense_tt_td_parallel(graph, key_nodes, time_attr="travel_time", dist_attr="length",
                               max_workers=2, chunksize=4, TT_filepath = "TT.dat", TD_filepath = "TD.dat"):
    key_nodes = list(key_nodes)
    n = len(key_nodes)
    node_to_idx = {node: i for i, node in enumerate(key_nodes)}

    BIG_T = np.float32(1e9)
    BIG_D = np.float32(1e9)

    TT = np.memmap(TT_filepath, dtype=np.float32, mode="w+", shape=(n, n))
    TD = np.memmap(TD_filepath, dtype=np.float32, mode="w+", shape=(n, n))
    TT[:] = BIG_T
    TD[:] = BIG_D

    tasks = [(i, u, node_to_idx, key_nodes, BIG_T, BIG_D) for i, u in enumerate(key_nodes)]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(graph, time_attr, dist_attr),
    ) as ex:
        futures = [ex.submit(compute_row, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="[Graph] Rows"):
            i, tt_row, td_row = fut.result()   # will re-raise worker exceptions here
            TT[i, :] = tt_row
            TD[i, :] = td_row

    TT.flush(); TD.flush()
    return node_to_idx, TT, TD

class travel_metrics:
    def __init__(self):
        self.node_to_idx=None
        self.TT = None
        self.TD = None
    
    def get_time(self, u, v):
        return self.TT[self.node_to_idx[u], self.node_to_idx[v]]
    
    def get_distance(self, u, v):
        return self.TD[self.node_to_idx[u], self.node_to_idx[v]]
    
    

    def build_dense_tt_td(self, graph:graph, instances, time_attr="travel_time", dist_attr="length", max_workers=2, chunksize=1, graph_name = ""):
        graph_name = graph_name
        ROOT = Path(__file__).resolve().parent
        DATA_DIR = Path(ROOT/"data")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = graph_name + "_nodes.npy"
        filepath = Path(DATA_DIR/filename)
        filepath_with_prefix = str(Path(DATA_DIR/graph_name))
        TT_filepath = filepath_with_prefix+"_TT.dat"
        TD_filepath = filepath_with_prefix+"_TD.dat"
        if filepath.exists():
            print(f"found a tavel_time file for {graph_name}, loading into memory")
            node_to_idx, TT, TD, meta = load_travel_cache_memmap(prefix=filepath_with_prefix)
            
        else:
            key_nodes = set()
            for inst in instances:
                for r in inst["requests"]:
                    if r.pickup:
                        key_nodes.add(r.pickup)
                    if r.dropoff:
                        key_nodes.add(r.dropoff)
                for v in inst["vehicles"]:
                    key_nodes.add(v.location)
                for s in inst.get("stations", []):
                    key_nodes.add(s.location)
            key_nodes = list(key_nodes)
            
            node_to_idx, TT, TD  = build_dense_tt_td_parallel(graph=graph, key_nodes=key_nodes, time_attr=time_attr, dist_attr=dist_attr, max_workers=max_workers, chunksize=chunksize, TD_filepath=TD_filepath, TT_filepath=TT_filepath)
            save_travel_cache_memmap(prefix=filepath_with_prefix, TT_memmap=TT, TD_memmap=TD, idx_to_node=node_to_idx, meta=graph_name)
        self.TT = TT
        self.TD = TD
        self.node_to_idx = node_to_idx
            
        return node_to_idx, TT, TD
    """BIG_T = np.float32(1e9)
    BIG_D = np.float32(1e9)
    key_nodes = set()
    for inst in instances:
        for r in inst["requests"]:
            if r.pickup:
                key_nodes.add(r.pickup)
            if r.dropoff:
                key_nodes.add(r.dropoff)
        for v in inst["vehicles"]:
            key_nodes.add(v.location)
        for s in inst.get("stations", []):
            key_nodes.add(s.location)
    key_nodes = list(key_nodes)
    n = len(key_nodes)
    node_to_idx = {node: i for i, node in enumerate(key_nodes)}

    TT = np.full((n, n), BIG_T, dtype=np.float32)
    TD = np.full((n, n), BIG_D, dtype=np.float32)

    for u in tqdm(key_nodes, desc="[Graph] Dijkstra sources"):
        uidx = node_to_idx[u]

        # Your custom Dijkstra returning {v: (t, d)}
        lengths = self.dijkstra_time_and_dist(G=graph, source=u, time_attr=time_attr, dist_attr=dist_attr)

        # Fill only columns we care about
        # (If you really want all N targets, this is still N assignments per row)
        for v, (t, d) in lengths.items():
            vidx = node_to_idx.get(v)
            if vidx is not None:
                TT[uidx, vidx] = np.float32(t)
                TD[uidx, vidx] = np.float32(d)

        # Ensure self-distance
        TT[uidx, uidx] = 0.0
        TD[uidx, uidx] = 0.0
        
    self.TT = TT
    self.TD = TD
    self.node_to_idx = node_to_idx
    
    return node_to_idx, TT, TD"""

    def build_tt_cache(self, graph, instances, time_attr="travel_time", dist_attr="length"):
        key_nodes = set()
        for inst in instances:
            for r in inst["requests"]:
                key_nodes.add(r.pickup)
                key_nodes.add(r.dropoff)
            for v in inst["vehicles"]:
                key_nodes.add(v.location)
            for s in inst.get("stations", []):
                key_nodes.add(s.location)

        key_nodes = list(key_nodes)
        n = len(key_nodes)
        print("unique key nodes:", n, "expected pairs:", n*n)

        tt = {}
        td = {}
        BIG = 1e9
        for u in tqdm(key_nodes, desc="[Graph] Dijkstra sources"):
            lengths = self.dijkstra_time_and_dist(graph, u, time_attr=time_attr, dist_attr=dist_attr)
            for v in key_nodes:
                t, d = lengths.get(v, (BIG, BIG))
                tt[(u, v)] = t
                td[(u, v)] = d

        return tt, td