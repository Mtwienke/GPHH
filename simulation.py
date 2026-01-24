# simulation.py — event-driven simulator with State + OSMnx graph routing
#
# This implements a sequential-decision simulation as requested:
# - State holds all mutable data (fleet, requests, event_queue, time, stations, graph)
# - Events: request_arrival, vehicle_free, (vehicle_charged placeholder)
# - Simulator.run consumes events from a min-heap, advancing State.time
# - Policies: π_veh used at request arrivals; π_req used when a vehicle becomes free
# - Travel times/distances come from an OSMnx/NetworkX graph via graph_utils.route_metrics

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from data import Vehicle, Request

from terminals import evaluate_policy_ast
from graph_utils import route_metrics


# -----------------------------
# Event & State abstractions
# -----------------------------

@dataclass(order=True)
class Event:
    time: float
    type: str = field(compare=False)  # 'request_arrival' | 'vehicle_free' | 'vehicle_charged'
    payload: Any = field(compare=False, default=None)


@dataclass
class State:
    travel_times: Any
    fleet: Dict[int, Vehicle]                 # vehicle_id -> Vehicle (expects: .id, .node, .charge_level, .capacity, .busy_until)
    requests: Dict[int, Request]              # request_id -> Request (expects: .id, .pickup_node, .dropoff_node, .arrival_time, .time_window)
    stations: List[Any]                   # list of charging stations (expects: .id, .node)
    time: float = 0.0
    event_queue: List[Event] = field(default_factory=list)
    waiting_ids: List[int] = field(default_factory=list)  # request ids waiting for assignment

    def push_event(self, ev: Event) -> None:
        heapq.heappush(self.event_queue, ev)

    def pop_event(self) -> Optional[Event]:
        if not self.event_queue:
            return None
        return heapq.heappop(self.event_queue)


# -----------------------------
# Simulator using (π_veh, π_req)
# -----------------------------

class Simulator:
    def __init__(self,
                travel_times,
                travel_distances,
                 service_time_pickup: float = 0.0,
                 service_time_dropoff: float = 0.0,
                 beta_energy_per_m: float = 0.0002,  # energy per meter
                 penalty_weight_late: float = 2.0,
                 log=None):
        self.t_pu = service_time_pickup
        self.t_do = service_time_dropoff
        self.beta = beta_energy_per_m
        self.pen_late = penalty_weight_late
        self.log = log  # optional Logger with .log(...)
        self.travel_times = travel_times
        self.travel_distances = travel_distances

    # -------------------------
    # Public entry point
    # -------------------------
    def run(self, instance: Dict[str, Any], policy_pair: Tuple[List[str], List[str]], travel_times) -> float:
        """
        instance must contain:
          - 'requests': List[Request] with attributes (id, pickup_node, dropoff_node, arrival_time, time_window)
          - 'vehicles': List[Vehicle] with attributes (id, node, charge_level, capacity)
          - 'stations': List[ChargingStation] with attribute (node)
        policy_pair = (veh_policy, req_policy)
        travel_times = NetworkX MultiDiGraph (OSMnx) with 'travel_time' on edges
        Returns objective value = total_travel_time + penalty_weight_late * pickup_lateness
        """
        # Deepcopy to avoid external mutation
        fleet = {v.id: deepcopy(v) for v in instance['vehicles']}
        reqs = {r.id: deepcopy(r) for r in instance['requests']}
        stations = deepcopy(instance['stations'])

        # Ensure vehicles have busy_until attribute
        for v in fleet.values():
            if not hasattr(v, 'busy_until'):
                v.busy_until = 0.0

        S = State(travel_times=travel_times, fleet=fleet, requests=reqs, stations=stations)

        # Seed event queue with request arrivals
        for r in S.requests.values():
            S.push_event(Event(time=float(r.arrival_time), type='request_arrival', payload=r.id))

        # Stats for objective
        total_travel_time = 0.0
        total_travel_dist = 0.0
        late_penalty = 0.0

        veh_policy, req_policy = policy_pair

        # Main event loop
        while True:
            ev = S.pop_event()
            if ev is None:
                break
            S.time = ev.time

            if ev.type == 'request_arrival':
                rid = ev.payload
                req = S.requests[rid]
                # Choose vehicle by π_veh among vehicles free at S.time
                best_vid, best_score = self._select_vehicle(S, req, veh_policy)
                if best_vid is None:
                    # Put request into waiting pool
                    S.waiting_ids.append(rid)
                    if self.log:
                        self.log.log(f"[t={S.time:.1f}] Request {rid} queued (no free vehicle)")
                else:
                    # Assign immediately and schedule completion
                    self._assign_and_schedule(S, best_vid, req, total_acc=(total_travel_time, total_travel_dist, late_penalty))
                    total_travel_time, total_travel_dist, late_penalty = S._last_totals  # see _assign_and_schedule

            elif ev.type == 'vehicle_free':
                vid = ev.payload
                v = S.fleet[vid]
                # If requests are waiting, select next by π_req
                if S.waiting_ids:
                    best_rid, best_score = self._select_request(S, v, req_policy)
                    if best_rid is not None:
                        req = S.requests[best_rid]
                        # Remove from waiting pool and assign starting at S.time
                        try:
                            S.waiting_ids.remove(best_rid)
                        except ValueError:
                            pass
                        self._assign_and_schedule(S, vid, req, start_time=S.time,
                                                   total_acc=(total_travel_time, total_travel_dist, late_penalty))
                        total_travel_time, total_travel_dist, late_penalty = S._last_totals
                        if self.log:
                            self.log.log(f"[t={S.time:.1f}] π_req: v{vid} -> r{best_rid}")
                # else stay idle

            elif ev.type == 'vehicle_charged':
                # Placeholder for future charging logic
                pass

        # Objective: travel time + penalty on lateness
        score = float(total_travel_time + self.pen_late * late_penalty)
        if 'normscore' in instance.keys():
            return score/instance['normscore']
        else:
            return score
    # -------------------------
    # Assignment helper
    # -------------------------
    def _assign_and_schedule(self, S: State, vid: int, req: Any, total_acc: Tuple[float, float, float], start_time: Optional[float] = None) -> None:
        """Assign vehicle vid to request, schedule its completion, and update totals.
        total_acc = (total_travel_time, total_travel_dist, late_penalty)
        If start_time is None, we assume assignment triggered by request_arrival at S.time.
        """
        total_travel_time, total_travel_dist, late_penalty = total_acc
        v = S.fleet[vid]
        now = S.time if start_time is None else start_time

        t_to_pu = self.travel_times[(v.location, req.pickup)]
        d_to_pu = self.travel_distances[(v.location, req.pickup)]
        t_pu_do = self.travel_times[(req.pickup, req.dropoff)]
        d_pu_do = self.travel_distances[(req.pickup, req.dropoff)]

        # Pickup can only start after arriving to pickup and not before earliest window
        start_service = max(now + t_to_pu, float(req.time_window[0]))
        pickup_finish = start_service + self.t_pu
        dropoff_finish = pickup_finish + t_pu_do + self.t_do

        # Lateness counted as arriving to pickup after latest window end
        lateness = max(0.0, (now + t_to_pu) - float(req.time_window[1]))
        late_penalty += lateness

        total_travel_time += t_to_pu + t_pu_do
        total_travel_dist += d_to_pu + d_pu_do

        # Update vehicle state and schedule completion
        v.location = req.dropoff
        v.busy_until = dropoff_finish
        v.charge_level -= self.beta * (d_to_pu + d_pu_do)
        S.push_event(Event(time=dropoff_finish, type='vehicle_free', payload=vid))

        if self.log:
            self.log.log(
                f"[t={S.time:.1f}] Assign v{vid}->r{req.id}; toPu={t_to_pu:.1f}s/{d_to_pu:.0f}m, pu->do={t_pu_do:.1f}s/{d_pu_do:.0f}m, finish={dropoff_finish:.1f}, late={lateness:.1f}")
        # store back
        S._last_totals = (total_travel_time, total_travel_dist, late_penalty)

    # -------------------------
    # Selection helpers
    # -------------------------
    def _select_vehicle(self, S: State, req, veh_policy) -> Tuple[Optional[int], float]:
        best_vid = None
        best_score = float('inf')
        for vid, v in S.fleet.items():
            if v.busy_until > S.time:
                continue
            score = evaluate_policy_ast(
                veh_policy,
                travel_times=self.travel_times,
                vehicle=v,
                request=req,
                requests=S.requests,
                vehicles=S.fleet,
                stations=S.stations,
                current_time=S.time,
            )
            if score < best_score:
                best_score = score
                best_vid = vid
        return best_vid, best_score

    def _select_request(self, S: State, v, req_policy) -> Tuple[Optional[int], float]:
        best_rid = None
        best_score = float('inf')
        for rid in list(S.waiting_ids):
            r = S.requests[rid]
            score = evaluate_policy_ast(
                req_policy,
                travel_times=self.travel_times,
                vehicle=v,
                request=r,
                requests=S.requests,
                vehicles=S.fleet,
                stations=S.stations,
                current_time=S.time,
            )
            if score < best_score:
                best_score = score
                best_rid = rid
        return best_rid, best_score
