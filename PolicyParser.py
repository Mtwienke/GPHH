from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import re
import pandas as pd
from graphviz import Digraph
import os

# --------- Tree model ----------
@dataclass
class Node:
    label: str
    left: Optional["Node"] = None
    right: Optional["Node"] = None

# --------- Parser for strings like OpNode(op='...', left=..., right=...) ----------
_TOKEN_OP = re.compile(r"OpNode\(op='([^']+)'\s*,\s*left=")
_TOKEN_TERM = re.compile(r"TerminalNode\(name='([^']+)'\)")

def _skip_ws_commas(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \n\t,":
        i += 1
    return i

def _parse_node(s: str, i: int = 0) -> Tuple[Node, int]:
    i = _skip_ws_commas(s, i)

    m = _TOKEN_TERM.match(s, i)
    if m:
        return Node(m.group(1)), m.end()

    m = _TOKEN_OP.match(s, i)
    if not m:
        raise ValueError(f"Cannot parse at pos {i}: {s[i:i+120]!r}")

    op = m.group(1)
    i = m.end()

    left, i = _parse_node(s, i)

    # find "right=" after left subtree
    key = "right="
    j = s.find(key, i)
    if j == -1:
        raise ValueError("Missing right= in OpNode(...)")
    i = j + len(key)

    right, i = _parse_node(s, i)

    # consume trailing ')'
    i = _skip_ws_commas(s, i)
    while i < len(s) and s[i] != ")":
        i += 1
    if i < len(s) and s[i] == ")":
        i += 1

    return Node(op, left, right), i

def parse_policy(policy_str: str) -> Node:
    node, _ = _parse_node(policy_str, 0)
    return node

# --------- Graphviz rendering ----------
def _add_to_dot(dot: Digraph, node: Node, prefix: str, counter: list[int]) -> str:
    nid = f"{prefix}{counter[0]}"
    counter[0] += 1

    # terminals vs ops: simple heuristic (ops are usually symbols/words like +, -, max, min, *)
    if node.left is None and node.right is None:
        dot.node(nid, node.label, shape="ellipse", style="filled", fillcolor="lightgray")
    else:
        dot.node(nid, node.label, shape="box")

    if node.left:
        lid = _add_to_dot(dot, node.left, prefix, counter)
        dot.edge(nid, lid)
    if node.right:
        rid = _add_to_dot(dot, node.right, prefix, counter)
        dot.edge(nid, rid)

    return nid

def render_two_policies_stacked_pdf(req_root: Node, veh_root: Node, out_pdf: Path,
                                   req_title: str = "Request policy", veh_title: str = "Vehicle policy"):
    """
    Creates ONE pdf with two subgraphs stacked (top: request, bottom: vehicle).
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    dot = Digraph(format="pdf")
    dot.attr(rankdir="TB")  # top-bottom

    # cluster for request
    with dot.subgraph(name="cluster_request") as c:
        c.attr(label=req_title, color="blue")
        _add_to_dot(c, req_root, prefix="r", counter=[0])

    # cluster for vehicle
    with dot.subgraph(name="cluster_vehicle") as c:
        c.attr(label=veh_title, color="darkgreen")
        _add_to_dot(c, veh_root, prefix="v", counter=[0])

    # Render. graphviz expects path without suffix; we provide stem.
    # Example: out_pdf = ".../X.pdf" -> render to ".../X.pdf"
    dot.render(str(out_pdf.with_suffix("")), cleanup=True)

# --------- Wrapper method ----------
def export_policy_trees_from_csv(csv_path: str | Path, out_dir: str | Path):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"city", "n_requests", "n_vehicles", "Request_Policy", "Vehicle_Policy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    for _, row in df.iterrows():
        city = str(row["city"])
        n_requests = int(row["n_requests"])
        n_vehicles = int(row["n_vehicles"])

        req_str = row["Request_Policy"]
        veh_str = row["Vehicle_Policy"]

        req_root = parse_policy(req_str)
        veh_root = parse_policy(veh_str)

        out_pdf = out_dir / f"{city}_n_r_{n_requests}_n_v_{n_vehicles}.pdf"

        render_two_policies_stacked_pdf(
            req_root, veh_root, out_pdf,
            req_title=f"Request policy ({city}, r={n_requests}, v={n_vehicles})",
            veh_title=f"Vehicle policy ({city}, r={n_requests}, v={n_vehicles})",
        )

os.environ["PATH"] += os.pathsep + r"C:\Users\Markus\Graphviz\bin"
export_policy_trees_from_csv(r"G:\Desktop\GPHH\Programm\GPHH\results_1\best_policies.csv", Path(r"G:\Desktop\GPHH\Programm\GPHH\results_1\summary") / "policy_trees_pdf")