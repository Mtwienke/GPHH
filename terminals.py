from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List, Union
import random
import math
import copy
import operator
import math
from data import travel_time, energy_cost, euclidean_distance, Vehicle, Request
import utils
import traceback

# Protected division
def protected_div(x, y):
	try:
		return x / y if y != 0 else 1.0
	except ZeroDivisionError:
		return 1.0
	
	
FUNCTIONS = {
	'+': operator.add,
	'-': operator.sub,
	'*': operator.mul,
	'/': protected_div,
	'min': min,
	'max': max,
}

def TVPU(v, r, graph, **kwargs):
    return travel_time(v.location, r.pickup, graph)

def COST(v, r, graph, **kwargs):
    return travel_time(v.location, r.pickup, graph) + travel_time(r.pickup, r.dropoff, graph)

def DEM(v, r, graph, **kwargs):
    return travel_time(v.location, r.pickup, graph) + travel_time(r.pickup, r.dropoff, graph)
    
def DUR(v, r, graph, **kwargs):
    return travel_time(r.pickup, r.dropoff, graph)
    
def SLACK(v, r, graph, t= None, **kwargs):
    return max(r.time_window[1] - t - travel_time(v.location, r.pickup, graph), 0)
    
def CAP(v, r, **kwargs):
    return v.capacity
    
def RT(v, r, **kwargs):
    return v.charge_level

def TVC(v, r, graph, stations=None, **kwargs):
    return min([travel_time(v.location, s.location, graph) for s in stations])
    
def AR(requests=None, **kwargs):
    return sum(requests)

def OBV(v, r, vehicles, graph, t=None, **kwargs):
    if t is None:
        raise ValueError("OBV needs current time t")

    best = float("inf")
    for veh in vehicles.values():
        if veh is v:
            continue

        # wann kann das Fahrzeug frühestens losfahren?
        available = max(t, veh.busy_until)

        # Fahrzeit zum Pickup (oder r.location, je nach Modell)
        tt = travel_time(veh.location, r.pickup, graph)

        # Wartezeit = busy-wait + anfahrtszeit
        wait = (available - t) + tt

        if wait < best:
            best = wait

    return best
     
def LCT(v, r, vehicles, graph, t=None, **kwargs):
    if t is None:
        raise ValueError("OBV needs current time t")

    best = float("inf")
    for veh in vehicles.values():

        # wann kann das Fahrzeug frühestens losfahren?
        available = max(t, veh.busy_until)

        # Fahrzeit zum Pickup (oder r.location, je nach Modell)
        tt = travel_time(veh.location, r.pickup, graph)

        # Wartezeit = busy-wait + anfahrtszeit
        wait = available + tt

        if wait < best:
            best = wait
    
    time_left = r.time_window[1] - best
    
    return time_left

def CRDV(v, r, vehicles, graph, radius_sec=300, **kwargs):
    """Count other vehicles within radius_sec travel time from v.location."""
    if vehicles is None:
        return 0

    count = 0
    for v2 in vehicles.values():
        if v2 is v:
            continue
        tt = travel_time(v.location, v2.location, graph)
        if tt <= radius_sec:
            count += 1
    return count

def CRPP(v, r, requests, graph, radius_sec=300, **kwargs):
    """Count other requests whose pickup is within radius_sec travel time from r.pickup."""
    if requests is None:
        return 0

    count = 0
    for r2 in requests.values():
        if r2 is r:
            continue
        tt = travel_time(r.pickup, r2.pickup, graph)
        if tt <= radius_sec:
            count += 1
    return count
    
def CRPD(v, r, requests, graph, radius_sec=300, **kwargs):
    """Count other requests whose pickup is within radius_sec travel time from r.pickup."""
    if requests is None:
        return 0

    count = 0
    for r2 in requests.values():
        if r2 is r:
            continue
        tt = travel_time(r.pickup, r2.dropoff, graph)
        if tt <= radius_sec:
            count += 1
    return count

def CRDP(v, r, requests, graph, radius_sec=300, **kwargs):
    """Count other requests whose pickup is within radius_sec travel time from r.pickup."""
    if requests is None:
        return 0

    count = 0
    for r2 in requests.values():
        if r2 is r:
            continue
        tt = travel_time(r.dropoff, r2.pickup, graph)
        if tt <= radius_sec:
            count += 1
    return count

def CRDD(v, r, requests, graph, radius_sec=300, **kwargs):
    """Count other requests whose pickup is within radius_sec travel time from r.pickup."""
    if requests is None:
        return 0

    count = 0
    for r2 in requests.values():
        if r2 is r:
            continue
        tt = travel_time(r.dropoff, r2.dropoff, graph)
        if tt <= radius_sec:
            count += 1
    return count

# Terminal feature functions
TERMINALS = {
	'TVPU': TVPU,
	'COST': COST,
	'DEM': DEM,
	'DUR': DUR,
	'SLACK': SLACK,
	'CAP': CAP,
	'RT': RT,
	'TVC': TVC,
    'AR': AR,
    'OBV': OBV,
    'LCT': LCT,
    'CRDV': CRDV,
    'CRPP': CRPP,
    'CRPD': CRPD,
    'CRDP': CRDP,
    'CRDD': CRDD,
}

Number = float

@dataclass(frozen=True)
class Node:
    def eval(self, ctx, FUNCTIONS, TERMINALS) -> Number:
        raise NotImplementedError

    def to_postfix(self) -> List[str]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class TerminalNode(Node):
    name: str  # nur Name, keine Funktion!

    def eval(self, ctx, FUNCTIONS, TERMINALS) -> Number:
        try:
            fn = TERMINALS[self.name]
            return float(fn(v=ctx["vehicle"], r=ctx["request"],
                            graph=ctx["graph"],
                            requests=ctx["requests"],
                            vehicles=ctx["vehicles"],
                            stations=ctx["stations"],
                            t=ctx["current_time"]))
        except Exception as e:
            print(f"in Terminal Node Exception with Terminal {self.name}: {e}")


    def to_postfix(self) -> List[str]:
        return [self.name]

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class ConstNode(Node):
    value: Number
    def eval(self, ctx, FUNCTIONS, TERMINALS) -> Number:
        return float(self.value)

    def to_postfix(self) -> List[str]:
        # Stabiler String
        return [repr(float(self.value))]

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class OpNode(Node):
    op: str      # nur Symbol
    left: Node
    right: Node

    def eval(self, ctx, FUNCTIONS, TERMINALS) -> Number:
        a = self.left.eval(ctx, FUNCTIONS, TERMINALS)
        b = self.right.eval(ctx, FUNCTIONS, TERMINALS)
        return float(FUNCTIONS[self.op](a, b))

    def to_postfix(self) -> List[str]:
        return self.left.to_postfix() + self.right.to_postfix() + [self.op]

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    
def evaluate_policy_ast(root: Node, *, graph, vehicle:Vehicle, request:Request, requests:Dict[int, Request], vehicles:Dict[int, Vehicle], stations, current_time) -> float:
    try:
        ctx = {
        "graph": graph,
        "vehicle": vehicle,
        "request": request,
        "requests": requests,
        "vehicles": vehicles,
        "stations": stations,
        "current_time": current_time,
        }
        val = root.eval(ctx, FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS)
        if not math.isfinite(val):
            print(f"not finite value: {val}")
            return float('inf')
        return val
    except Exception as e:
        print(f"in Exception of evalue_policy_ast: {e}")
        #print (f"evaluating {ctx}")
        # Optional: Logging hier
        return float('inf')
    
def random_terminal(TERMINALS: Dict[str, Callable[..., Number]], p_const=0.2,
                    const_range=(-5.0, 5.0)) -> Node:
    if random.random() < p_const:
        return ConstNode(random.uniform(*const_range))
    name = random.choice(list(TERMINALS.keys()))
    return TerminalNode(name=name)

def generate_norming_policy(FUNTIONS = FUNCTIONS, TERMINALS = TERMINALS):
    norming_policy = (TerminalNode("TVPU"), TerminalNode("TVPU"))
    return norming_policy

def random_tree(max_depth: int = 1,
                p_const=0.2,
                const_range=(-5.0, 5.0),
                p_terminal_at_depth: float = 0.3, 
                FUNCTIONS: Dict[str, Callable[[Number, Number], Number]] = FUNCTIONS,
                TERMINALS: Dict[str, Callable[..., Number]] = TERMINALS,
                ) -> Node:
    # Wenn max_depth == 0 -> Terminal erzwingen
    if max_depth <= 0 or random.random() < p_terminal_at_depth:
        return random_terminal(TERMINALS, p_const=p_const, const_range=const_range)

    op = random.choice(list(FUNCTIONS.keys()))
    left = random_tree(FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, max_depth=max_depth-1,
                       p_const=p_const, const_range=const_range,
                       p_terminal_at_depth=p_terminal_at_depth)
    right = random_tree(FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS, max_depth=max_depth-1,
                        p_const=p_const, const_range=const_range,
                        p_terminal_at_depth=p_terminal_at_depth)
    return OpNode(op=op, left=left, right=right)

Path = Tuple[str, ...]  # Sequenz aus "L" / "R"

def all_paths(root: Node) -> List[Path]:
    paths: List[Path] = [()]
    if isinstance(root, OpNode):
        for p in all_paths(root.left):
            paths.append(("L",) + p)
        for p in all_paths(root.right):
            paths.append(("R",) + p)
    return paths

def get_subtree(root: Node, path: Path) -> Node:
    node = root
    for step in path:
        if not isinstance(node, OpNode):
            raise ValueError("Path goes through non-OpNode")
        node = node.left if step == "L" else node.right
    return node

def replace_subtree(root: Node, path: Path, new_sub: Node) -> Node:
    if path == ():
        return new_sub
    if not isinstance(root, OpNode):
        raise ValueError("Cannot replace below a terminal")
    step = path[0]
    rest = path[1:]
    if step == "L":
        return OpNode(op=root.op,
                      left=replace_subtree(root.left, rest, new_sub),
                      right=root.right)
    else:
        return OpNode(op=root.op,
                      left=root.left,
                      right=replace_subtree(root.right, rest, new_sub))
        
def subtree_crossover_ast(p1: Node, p2: Node) -> Tuple[Node, Node]:
    paths1 = all_paths(p1)
    paths2 = all_paths(p2)

    cut1 = random.choice(paths1)
    cut2 = random.choice(paths2)

    sub1 = get_subtree(p1, cut1)
    sub2 = get_subtree(p2, cut2)

    c1 = replace_subtree(p1, cut1, sub2)
    c2 = replace_subtree(p2, cut2, sub1)
    return c1, c2

def subtree_mutation_ast(root: Node,
                         FUNCTIONS = FUNCTIONS,
                         TERMINALS = TERMINALS,
                         max_init_depth: int = 3,
                         max_tree_depth: int = 6,
                         p_const: float = 0.2,
                         const_range=(-5.0, 5.0),
                         retries: int = 20) -> Node:
    """
    Wählt einen zufälligen Knoten (Subtree) und ersetzt ihn durch einen neuen zufälligen Subtree.
    Mit retries, um max_tree_depth einzuhalten.
    """
    paths = all_paths(root)
    cut = random.choice(paths)

    for _ in range(retries):
        new_sub = random_tree(FUNCTIONS=FUNCTIONS, TERMINALS=TERMINALS,
                              max_depth=max_init_depth,
                              p_const=p_const,
                              const_range=const_range)

        mutated = replace_subtree(root, cut, new_sub)

        if depth(mutated) <= max_tree_depth:
            return mutated

    # Falls wir kein gültiges/kleines Exemplar finden, gib unverändert zurück
    return root

def depth(root: Node) -> int:
    if not isinstance(root, OpNode):
        return 0
    return 1 + max(depth(root.left), depth(root.right))