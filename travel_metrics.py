from math import inf
import heapq
import numpy as np
from tqdm import tqdm

class travel_metrics:
    def __init__(self):
        self.node_to_idx=None
        self.TT = None
        self.TD = None
    
    def get_time(self, u, v):
        return self.TT[self.node_to_idx[u], self.node_to_idx[v]]
    
    def get_distance(self, u, v):
        return self.TD[self.node_to_idx[u], self.node_to_idx[v]]
    
    def dijkstra_time_and_dist(self, G, source, time_attr="travel_time", dist_attr="length"):
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

    def build_dense_tt_td(self, graph, instances, time_attr="travel_time", dist_attr="length"):
        BIG_T = np.float32(1e9)
        BIG_D = np.float32(1e9)
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
        node_to_idx = {node: i for i, node in enumerate(key_nodes)}

        TT = np.full((n, n), BIG_T, dtype=np.float32)
        TD = np.full((n, n), BIG_D, dtype=np.float32)

        for u in tqdm(key_nodes, desc="[Graph] Dijkstra sources"):
            uidx = node_to_idx[u]

            # Your custom Dijkstra returning {v: (t, d)}
            lengths = self.dijkstra_time_and_dist(graph, u, time_attr=time_attr, dist_attr=dist_attr)

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
        
        return node_to_idx, TT, TD

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