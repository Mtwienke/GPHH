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
from data import Vehicle, Request, Request_Type

from terminals import evaluate_policy_ast, choose_closest_station, protected_div, choose_closest_station_from_location
from travel_metrics import travel_metrics

from vehicle_recorder import ActivityRecorder, VehState, SoCPoint, StateSegment, export_csv, plot_vehicle_png
from request_recorder import RequestRecorder, ReqState, RequestSegment, export_dataclasses_csv, plot_request_segments_png
from pathlib import Path
from datetime import datetime

# -----------------------------
# Event & State abstractions
# -----------------------------

@dataclass(order=True)
class Event:
    time: float
    type: str = field(compare=False)  # 'request_arrival' | 'vehicle_free' | 'vehicle_charged' | 'request_timeout'
    payload: Any = field(compare=False, default=None)


@dataclass
class State:
    travel_metrics: travel_metrics
    fleet: Dict[int, Vehicle]                 # vehicle_id -> Vehicle (expects: .id, .node, .charge_level, .capacity, .busy_until)
    requests: Dict[int, Request]              # request_id -> Request (expects: .id, .pickup_node, .dropoff_node, .arrival_time, .time_window)
    stations: List[Any]                   # list of charging stations (expects: .id, .node)
    time: float = 0.0
    event_queue: List[Event] = field(default_factory=list)
    waiting_ids: List[int] = field(default_factory=list)  # request ids waiting for assignment
    _last_totals: Tuple[float, float, float, float] = (0, 0, 0, 0)

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
                 travel_metrics,
                 service_time_pickup: float = 0.0,
                 service_time_dropoff: float = 0.0,
                 beta_energy_per_m: float = 0.001,  # energy per meter
                 penalty_weight_late: float = 30*60*10,
                 gamma_charging: float = 0.05, # energy charged per second
                 log=None):
        self.t_pu = service_time_pickup
        self.t_do = service_time_dropoff
        self.beta = beta_energy_per_m
        self.pen_late = penalty_weight_late
        self.log = log  # optional Logger with .log(...)
        self.travel_metrics = travel_metrics
        self.charging = gamma_charging

    # -------------------------
    # Public entry point
    # -------------------------
    def run(self, instance: Dict[str, Any], policy_pair: Tuple[List[str], List[str]], travel_metrics=None, plot_dir = None) -> float:
        """
        instance must contain:
          - 'requests': List[Request] with attributes (id, pickup_node, dropoff_node, arrival_time, time_window)
          - 'vehicles': List[Vehicle] with attributes (id, node, charge_level, capacity)
          - 'stations': List[ChargingStation] with attribute (node)
        policy_pair = (veh_policy, req_policy)
        travel_metrics = NetworkX MultiDiGraph (OSMnx) with 'travel_time' on edges
        Returns objective value = total_travel_time + penalty_weight_late * pickup_lateness
        """
        if travel_metrics == None:
            travel_metrics = self.travel_metrics
        # Deepcopy to avoid external mutation
        fleet = {v.id: deepcopy(v) for v in instance['vehicles']}
        reqs = {r.id: deepcopy(r) for r in instance['requests']}
        stations = deepcopy(instance['stations'])
        
        rec = ActivityRecorder()

        #print(f"n_requests = {len(reqs)}")

        # initial state: alle idle bei t=0 (oder S.time initial)
        for v in fleet.values():
            rec.start(v.id, t=0.0, state=VehState.IDLE,
                    soc=v.charge_level / v.battery_capacity)

        # Ensure vehicles have busy_until attribute
        for v in fleet.values():
            if not hasattr(v, 'busy_until'):
                v.busy_until = 0.0

        S = State(travel_metrics=travel_metrics, fleet=fleet, requests=reqs, stations=stations)
        S.rec = rec
        S.req_rec = RequestRecorder(ignore_types={Request_Type.CHARGE, Request_Type.IDLE}, ignore_ids={-1, -2})

        # Seed event queue with request arrivals
        for r in S.requests.values():
            if r.arrival_time:
                S.push_event(Event(time=float(r.arrival_time), type='request_arrival', payload=r.id))

        # Stats for objective
        total_travel_time = 0.0
        total_travel_dist = 0.0
        total_waittime = 0.0
        late_penalty = 0.0

        veh_policy, req_policy = policy_pair
        
        #print(f"initial event queue size of {len(S.event_queue)}")

        # Main event loop
        while True:
            ev = S.pop_event()
            #print(f"event is {ev}")
            
            if ev is None:
                break
            S.time = ev.time
            
            if ev.type == 'request_timeout':
                rid = ev.payload
                #print(f"req {rid} timed out")
                if rid in S.waiting_ids:
                    S.req_rec.switch(req, S.time, ReqState.LOST, vehicle_id=None)
                    S.waiting_ids.remove(rid)
                    total_travel_time, total_travel_dist, late_penalty, total_waittime = S._last_totals
                    S._last_totals = total_travel_time, total_travel_dist, late_penalty + 1, total_waittime
                continue

            elif ev.type == 'request_arrival':
                rid = ev.payload
                req = S.requests[rid]
                S.waiting_ids.append(rid)
                # Choose vehicle by π_veh among vehicles free at S.time
                best_vid, best_score = self._select_vehicle(S, req, veh_policy)
                if best_vid is None:
                    #print(f"Request {rid} got no fitting vehicle, putting in waiting")
                    # Leave request in waiting pool
                    S.req_rec.start_waiting(req, S.time)
                    S.push_event(Event(time=S.time+(30*60), type='request_timeout', payload=rid))
                    if self.log:
                        self.log.log(f"[t={S.time:.1f}] Request {rid} queued (no free vehicle)")
                elif best_vid < 0:
                    S.req_rec.start_waiting(req, S.time)
                    S.push_event(Event(time=S.time+(30*60), type='request_timeout', payload=rid))
                    if self.log:
                        self.log.log(f"[t={S.time:.1f}] Request {rid} queued (no free vehicle)")
                else:
                    # Assign immediately and schedule completion
                    #print(f"assigning request {rid}")
                    self._assign_and_schedule(S, best_vid, req)
                    total_travel_time, total_travel_dist, late_penalty, total_waittime = S._last_totals  # see _assign_and_schedule

            elif ev.type == 'vehicle_free':
                vid = ev.payload
                v = S.fleet[vid]
                # If requests are waiting, select next by π_req
                if S.waiting_ids:
                    best_rid, best_score = self._select_request(S, v, req_policy)
                    if best_rid is not None:
                        req = S.requests[best_rid]
                        # Remove from waiting pool and assign starting at S.time
                        if req.request_type == Request_Type.IDLE:
                            rec.switch(vid, S.time, VehState.IDLE, v.charge_level / v.battery_capacity)
                            #print("Staying idle was best")
                            continue
                        elif req.request_type == Request_Type.CHARGE:
                            self._charge_vehicle(S=S, vid=vid, req=req)
                            #print("Charging was best")
                        else:
                            self._assign_and_schedule(S, vid, req, start_time=S.time)
                            total_travel_time, total_travel_dist, late_penalty, total_waittime = S._last_totals
                            if self.log:
                                self.log.log(f"[t={S.time:.1f}] π_req: v{vid} -> r{best_rid}")
                else:
                    rec.switch(vid, S.time, VehState.IDLE, v.charge_level / v.battery_capacity)

        # Objective: travel time + penalty on lateness
        score = float(0.12*total_travel_dist + total_waittime + self.pen_late * late_penalty)
        rec.close_all(t_end=S.time, fleet=S.fleet)
        S.req_rec.close_all(S.time)
        if plot_dir:
            PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
            LOG_ROOT = PROJECT_ROOT / "logs"
            timestamp = datetime.now().strftime("[%Y-%m-%d %H-%M]")
            gen_dir = safe_subdir(LOG_ROOT, timestamp, "validation", "plots", plot_dir)
            gen_dir.mkdir(parents=True, exist_ok=True)
            export_csv(rec, gen_dir/"segments.csv", gen_dir / "soc.csv")
            export_dataclasses_csv(S.req_rec.segments, gen_dir/"req_segments.csv")
            plot_vehicle_png(gen_dir/"segments.csv", gen_dir / "soc.csv", gen_dir)
            plot_request_segments_png(gen_dir/"req_segments.csv")
        if 'normscore' in instance.keys():
            normed_score = protected_div(score, instance['normscore'])
        else:
            normed_score = None
        if score == 0:
            print("Got a 0")
        return score, normed_score
    # -------------------------
    # Assignment helper
    # -------------------------
    def _assign_and_schedule(self, S: State, vid: int, req: Any, start_time: Optional[float] = None) -> None:
        """Assign vehicle vid to request, schedule its completion, and update totals.
        total_acc = (total_travel_time, total_travel_dist, late_penalty)
        If start_time is None, we assume assignment triggered by request_arrival at S.time.
        """
        rec = S.rec
        total_travel_time, total_travel_dist, late_penalty, total_waittime = S._last_totals
        v = S.fleet[vid]
        now = S.time if start_time is None else start_time
        

        t_to_pu = self.travel_metrics.get_time(v.location, req.pickup)
        d_to_pu = self.travel_metrics.get_distance(v.location, req.pickup)
        t_pu_do = self.travel_metrics.get_time(req.pickup, req.dropoff)
        d_pu_do = self.travel_metrics.get_distance(req.pickup, req.dropoff)
        



        # Pickup can only start after arriving to pickup and not before earliest window
        start_service = max(now + t_to_pu, float(req.time_window[0]))
        total_waittime = total_waittime + start_service - req.time_window[0]
        pickup_finish = start_service + self.t_pu
        dropoff_finish = pickup_finish + t_pu_do + self.t_do
        dropoff_charging_station, _ = choose_closest_station_from_location(req.dropoff, S.stations, self.travel_metrics)
        d_do_cs = self.travel_metrics.get_distance(req.dropoff, dropoff_charging_station.location)
        future_distance_to_charging = d_to_pu + d_pu_do + d_do_cs

        # Lateness counted as arriving to pickup after latest window end
        if (now + t_to_pu) - float(req.time_window[1]) > 0:
            #print(f"too late: taking {t_to_pu} seconds to pickup, only having {req.time_window[1]-now} seconds from {now} until {req.time_window[1]}. Time window is {req.time_window[0]}, {req.time_window[1]}")
            #print(f"As a test: (now + t_to_pu) - float(req.time_window[1]) is ({now + t_to_pu}) - {float(req.time_window[1])} {(now + t_to_pu) - float(req.time_window[1])}")
            late_penalty += 1
            S.req_rec.switch(req, S.time, ReqState.LOST, vehicle_id=None)
        
        elif v.charge_level-(d_to_pu+d_pu_do+future_distance_to_charging)*self.beta < 0:
            #print(f"Vehicle {vid} was too low on battery (charge of {v.charge_level}), charging...")
            self._charge_vehicle(S=S, vid=vid) 
            S.push_event(Event(time=S.time+(30*60), type='request_timeout', payload=req.id))
            return
        
        else:
            try:
                S.waiting_ids.remove(req.id)
            except ValueError:
                print(f"tried removing request id {req.id}, but wasnt successful")
            #print(f"assigning for real")
            total_travel_time += t_to_pu + t_pu_do
            total_travel_dist += d_to_pu + d_pu_do
            rec.switch(vid, now, VehState.DRIVE_TO_PICKUP, v.charge_level / v.battery_capacity, request_id=req.id)
            S.req_rec.switch(req, S.time, ReqState.ASSIGNED, vehicle_id=vid)
            v.charge_level -= self.beta * d_to_pu
            rec.switch(vid, start_service, VehState.FULFILL_REQUEST, v.charge_level / v.battery_capacity, request_id=req.id)
            S.req_rec.switch(req, start_service, ReqState.IN_SERVICE, vehicle_id=vid)
            # Update vehicle state and schedule completion
            v.location = req.dropoff
            v.busy_until = dropoff_finish
            v.charge_level -= self.beta * d_pu_do
            rec.switch(vid, dropoff_finish, VehState.IDLE, v.charge_level / v.battery_capacity)
            S.req_rec.switch(req, dropoff_finish, ReqState.COMPLETED, vehicle_id=vid)
            S.push_event(Event(time=dropoff_finish, type='vehicle_free', payload=vid))

            if self.log:
                self.log.log(
                    f"[t={S.time:.1f}] Assign v{vid}->r{req.id}; toPu={t_to_pu:.1f}s/{d_to_pu:.0f}m, pu->do={t_pu_do:.1f}s/{d_pu_do:.0f}m, finish={dropoff_finish:.1f}, late={lateness:.1f}")
            # store back
        S._last_totals = (total_travel_time, total_travel_dist, late_penalty, total_waittime)

    def _charge_vehicle(self, S: State, vid: int, req: Any = None, start_time: Optional[float] = None) -> None:
        total_travel_time, total_travel_dist, late_penalty, total_waittime = S._last_totals
        v = S.fleet[vid]
        if v.charge_level == v.battery_capacity:
            #print(f"Vehicle {vid} already fully charged, putting to idle")
            S.rec.switch(vid, S.time, VehState.IDLE, v.charge_level / v.battery_capacity)
            return
        now = S.time if start_time is None else start_time
        
        rec = S.rec
        rec.switch(vid, now, VehState.DRIVE_TO_PICKUP, v.charge_level / v.battery_capacity) 
        nearest_station, nstt = choose_closest_station(vehicle=v, stations=S.stations, travel_metrics=self.travel_metrics)
        nstd = self.travel_metrics.get_distance(v.location, nearest_station.location)
        
        v.charge_level -= nstd * self.beta
        rec.switch(vid, now+nstt, VehState.CHARGING, v.charge_level / v.battery_capacity, station_id=nearest_station.id)      
        charging_time = (v.battery_capacity-v.charge_level)/self.charging
        v.charge_level = v.battery_capacity
        v.location = nearest_station.location
        v.busy_until = now + nstt + charging_time
        rec.switch(vid, v.busy_until, VehState.IDLE, v.charge_level / v.battery_capacity)
        S.push_event(Event(time=v.busy_until, type='vehicle_free', payload=vid))
        S._last_totals = (total_travel_time+nstt, total_travel_dist+nstd, late_penalty, total_waittime)
        
    # -------------------------
    # Selection helpers
    # -------------------------
    def _select_vehicle(self, S: State, req, veh_policy) -> Tuple[Optional[int], float]:
        best_vid = None
        best_score = float('inf')
        for vid, v in S.fleet.items():
            if v.busy_until > S.time:
                continue
            if vid < 0:
                v.location = req.pickup
            score = evaluate_policy_ast(
                veh_policy,
                travel_metrics=self.travel_metrics,
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
                travel_metrics=self.travel_metrics,
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

def safe_subdir(base: Path, *parts: str) -> Path:
    """
    Erzeugt base/parts und verhindert, dass der Pfad base verlässt.
    """
    base = base.resolve()
    out = (base.joinpath(*parts)).resolve()
    if base not in out.parents and out != base:
        raise ValueError(f"Unsafe path: {out} is not inside {base}")
    return out