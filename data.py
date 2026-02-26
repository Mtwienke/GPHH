#!/usr/bin/env python3

# data.py
from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict, Union
from travel_metrics import travel_metrics
from datetime import datetime
from enum import Enum, auto
import numpy
from pathlib import Path
import json
from GraphManager import GraphManager
from os import listdir
from tqdm import tqdm
#from graph_utils import route_metrics

Coord= Tuple[float, float]

class Request_Type(Enum):
	RIDE = auto()
	CHARGE = auto()
	IDLE = auto()

@dataclass
class Request:
    id: int
    request_type: Request_Type

    pickup: Optional[Coord] = None
    dropoff: Optional[Coord] = None
    arrival_time: float = 0.0
    time_window: Optional[Tuple[float, float]] = None
    demand: int = 0  # Ride: >0, Charge/Idle: 0
	
@dataclass
class Vehicle:
	id: int
	location: Optional[Tuple[float, float]]
	schedule: List[Request] = field(default_factory=list)
	capacity: int = 3
	battery_capacity: float = 200.0
	charge_level: float = 200.0
	busy_until: float = 0.0
	
	
@dataclass
class ChargingStation:
	id: int
	location: Tuple[float, float]
	charging_rate: float = 0.05
	
	
def euclidean_distance(p1, p2):
	return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def travel_time(u, v, tm: travel_metrics, phi=0.01, t_service=0):
	travelTime = tm.get_time(u, v)
	return travelTime


def energy_cost(u, v, tm:travel_metrics, beta=0.0005):
	return beta * tm.get_distance(u, v)

def generate_instances(graph, n_instances=10, n_requests=50, n_vehicles = 5, time_horizon=60*60*24, n_chargingStations = 3, simulator=None, norming_policy = None):
	instances = []
	nodes = [n for n, d in graph.nodes(data=True) if d.get("building")]
	range_instances = tqdm(range(n_instances), "Generating instances")
	for i in range_instances:
		requests = generate_requests(n_requests=n_requests, nodes=nodes, time_horizon=time_horizon)
			
		vehicles = [
			Vehicle(id=k, location=random.choice(nodes))
			for k in range(n_vehicles)
		]
		
		stations = [
			ChargingStation(id=s, location=random.sample(nodes, n_chargingStations)[s])
			for s in range(n_chargingStations)
		]
		instance = {
			'requests': requests,
			'vehicles': vehicles,
			'stations': stations
		}
		if simulator and norming_policy:
			print(f"Running instance with {norming_policy} for score to norm against.")
			norm_score = simulator.run(instance=instance, policy_pair=norming_policy, graph=graph)
			instance['normscore']=norm_score
		instances.append(instance)
	return instances

def import_instances(graphManager: GraphManager, ext_req_instances: List[List[ExternalRequest]], n_vehicles = 5, time_horizon=60*60*24, n_chargingStations = 3, simulator=None, norming_policy = None):
	instances = []
	for external_requests in ext_req_instances:
		requests = []
		pickup_nodes = []
		for req in external_requests:
			pickup = graphManager.get_nearest_node(req.pickup[0], req.pickup[1])
			pickup_nodes.append(pickup)
			dropoff = graphManager.get_nearest_node(req.dropoff[0], req.dropoff[1])
			if req.pickup_time:
				requests.append(Request(id=req.id,
					request_type=req.request_type,
					pickup=pickup,     
					dropoff=dropoff,   
					arrival_time=req.pickup_time,
					time_window=(req.pickup_time, req.pickup_time+30*60),
					demand=req.demand
				))
		vehicle_location = graphManager.middle_node_from_nodes(pickup_nodes)
		vehicles = [
				Vehicle(id=k, location=vehicle_location)
				for k in range(n_vehicles)
			]
		stations = [
				ChargingStation(id=s, location=vehicle_location)
				for s in range(n_chargingStations)
			]
		instance = {
				'requests': requests,
				'vehicles': vehicles,
				'stations': stations
			}
		if simulator and norming_policy:
				print(f"Running instance with {norming_policy} for score to norm against.")
				norm_score = simulator.run(instance=instance, policy_pair=norming_policy, graph=graphManager.graph)
				instance['normscore']=norm_score
		instances.append(instance)
	return instances

def generate_requests(n_requests, nodes, time_horizon, full_random=False, work_percent = 50):
	requests = []
	if full_random:
		for r in range(n_requests):
			startNode, destinationNode = random.sample(nodes, 2)
			arrival_time = random.uniform(0, time_horizon - 180)
			tstart = arrival_time
			tend = arrival_time + 30*60
			requests.append(Request(
				request_type=Request_Type.RIDE,
				id=r,
				pickup=startNode,
				dropoff=destinationNode,
				arrival_time=arrival_time,
				time_window=(tstart, tend),
			))
	else:
		for r in range(int(work_percent*n_requests/200)):
			perm = nodes[:]
			random.shuffle(perm)
			homeNodes, workNodes = perm[int(len(nodes)/3):], perm[:int(len(nodes)/3)]
			workNode = random.choice(workNodes)
			homeNode = random.choice(homeNodes)
			workway_starttime = numpy.random.normal(7.5*60*60, 30*60)
			requests.append(Request(
				request_type=Request_Type.RIDE,
				id=2*r,
				pickup=homeNode,
				dropoff=workNode,
				arrival_time=workway_starttime,
				time_window=(workway_starttime, workway_starttime+30*60)
			))
			requests.append(Request(
				request_type=Request_Type.RIDE,
				id=2*r+1,
				pickup=workNode,
				dropoff=homeNode,
				arrival_time=workway_starttime+9*60*60,
				time_window=(workway_starttime+9*60*60, workway_starttime+9.50*60*60)
			))
		for r in range(len(requests), n_requests):
			startNode, destinationNode = random.sample(nodes, 2)
			arrival_time = random.uniform(0, time_horizon - 180)
			tstart = arrival_time
			tend = arrival_time + 30*60
			requests.append(Request(
				request_type=Request_Type.RIDE,
				id=r,
				pickup=startNode,
				dropoff=destinationNode,
				arrival_time=arrival_time,
				time_window=(tstart, tend),
			))

	requests.append(Request(
		request_type=Request_Type.CHARGE,
		id=-1,
		pickup=None,
		dropoff=None,
		arrival_time=None,
		time_window=None,
		#max_ride_time=max_ride_time
	))
	requests.append(Request(
		request_type=Request_Type.IDLE,
		id=-2,
		pickup=None,
		dropoff=None,
		arrival_time=None,
		time_window=None,
		#max_ride_time=max_ride_time
	))
	return requests


from datetime import datetime, timedelta

ONLINE_WINDOW_SEC = 30 * 60

def _parse_dt(s: str | None) -> float | None:
    time_in_seconds = None
    if s:
        t = datetime.fromisoformat(s)
        time_in_seconds = (t.hour * 60 + t.minute) * 60 + t.second
    return time_in_seconds

def compute_arrival_and_window(rp: dict, pickup_node, dropoff_node, travel_metrics, reference_time: datetime):
    pickupTime = rp.get("pickupTime")
    targetTime = rp.get("targetTime")

    if pickupTime is not None:
        pu_start_dt = _parse_dt(pickupTime.get("StartTime"))
        if pu_start_dt is None:
            raise ValueError("pickupTime present but StartTime missing/null")

        arrival_dt = pu_start_dt
        tw_start_dt = pu_start_dt
        tw_end_dt = pu_start_dt + timedelta(seconds=ONLINE_WINDOW_SEC)

    else:
        # pickupTime is null -> targetTime must exist
        if targetTime is None:
            raise ValueError("Both pickupTime and targetTime are null (unexpected).")

        target_end_dt = _parse_dt(targetTime.get("EndTime"))
        if target_end_dt is None:
            raise ValueError("targetTime present but EndTime missing/null")

        t_trip = float(travel_metrics.get_time(pickup_node, dropoff_node))  # seconds
        latest_pickup_dt = target_end_dt - timedelta(seconds=t_trip)

        tw_end_dt = latest_pickup_dt
        tw_start_dt = latest_pickup_dt - timedelta(seconds=ONLINE_WINDOW_SEC)
        arrival_dt = tw_start_dt  # request becomes known at window start

    arrival_time = (arrival_dt - reference_time).total_seconds()
    tw_start = (tw_start_dt - reference_time).total_seconds()
    tw_end = (tw_end_dt - reference_time).total_seconds()
    return float(arrival_time), (float(tw_start), float(tw_end))

def load_external_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
    
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

#Coord = Tuple[float, float]  # (lat, lon)

@dataclass
class ExternalRequest(Request):
    id: int
    request_type: Request_Type
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    pickup: Optional[Coord] = None
    dropoff: Optional[Coord] = None
    arrival_time: float = 0.0
    time_window: Optional[Tuple[float, float]] = None
    demand: int = 0  # Ride: >0, Charge/Idle: 0


def import_requests_from_external_json(
    external: Dict[str, Any],
    *,
    reference_time: Optional[datetime] = None,
    return_coords: bool = True,
) -> Tuple[List[Coord], List[Request], datetime]:
    """
    Returns (coords, requests, reference_time)

    coords: list of (lat, lon) pickup+dropoff (optional)
    requests: list of Request objects (pickup/dropoff as node ids)
    reference_time: datetime used as t=0
    """
    steps = external.get("simulationSteps", [])
    coords: List[Coord] = []
    requests: List[Request] = []

    if not steps:
        return coords, requests, reference_time or datetime.now().astimezone()

    """# reference_time (t=0) wählen: earliest pickupTime.StartTime oder targetTime.EndTime oder SimulatedTime
    if reference_time is None:
        candidates: List[datetime] = []
        for st in steps:
            rp = st.get("requestParameter") or {}
            pu = rp.get("pickupTime") or {}
            tgt = rp.get("targetTime") or {}

            dt = (
                _parse_dt(pu.get("StartTime"))
                or _parse_dt(tgt.get("EndTime"))
                or _parse_dt(st.get("SimulatedTime"))
            )
            if dt is not None:
                candidates.append(dt)
        reference_time = min(candidates) if candidates else datetime.now().astimezone()"""

    for idx, st in enumerate(steps):
        rp = st.get("requestParameter") or {}
        pickupTime = rp.get("pickupTime")
        targetTime = rp.get("targetTime")
        rp = st.get("requestParameter") or {}

        cur = rp.get("CurrentLocation") or {}
        tgt = rp.get("TargetLocation") or {}
        if "Latitude" not in cur or "Longitude" not in cur or "Latitude" not in tgt or "Longitude" not in tgt:
            continue

        pickup_coord = (float(cur["Latitude"]), float(cur["Longitude"]))
        dropoff_coord = (float(tgt["Latitude"]), float(tgt["Longitude"]))

        if return_coords:
            coords.append(pickup_coord)
            coords.append(dropoff_coord)

        # demand
        adults = int(rp.get("requestedAdults") or 0)
        childs = int(rp.get("requestedChilds") or 0)
        demand = max(1, adults + childs)

        # id
        user_guid = rp.get("UserGuid")
        try:
            rid = int(user_guid)
        except Exception:
            rid = idx
            
        req = ExternalRequest(id=rid,
            request_type=Request_Type.RIDE,
            pickup=pickup_coord,     
            dropoff=dropoff_coord,   
            arrival_time=None,
            time_window=None,
            demand=demand)
        if pickupTime is not None:
            pu_start_dt = _parse_dt(pickupTime.get("StartTime"))
            if pu_start_dt is None:
                raise ValueError("pickupTime present but StartTime missing/null")
            req.pickup_time = pu_start_dt
            #print(f"Pickup time is: {req.pickup_time}")
        else:
        	# pickupTime is null -> targetTime must exist
            if targetTime is None:
                raise ValueError("Both pickupTime and targetTime are null (unexpected).")
            target_end_dt = _parse_dt(targetTime.get("EndTime"))
            if target_end_dt is None:
                raise ValueError("targetTime present but EndTime missing/null")
            req.dropoff_time = target_end_dt
        requests.append(req)

    return coords, requests, reference_time


def import_ext_instances(place: str):
	PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
	EXT_DATA_ROOT = PROJECT_ROOT / "external_data"
	if place == "Höxter, Germany":
		folderpath = EXT_DATA_ROOT / "hx_core_avg_max"
	elif place == "Paderborn, Germany":
		folderpath = EXT_DATA_ROOT / "pb_nachfragen"
	else:
		print(f"No external data for {place}, skipping adding real instances")
		return
	jsons = []
	for file in listdir(folderpath):
		json = load_external_json(folderpath/file)
		jsons.append(json)
	request_instances = []
	request_coords = []
	for json in jsons:
		requests_nodes, requests, _ = import_requests_from_external_json(json)
		request_instances.append(requests)
		request_coords.extend(requests_nodes)
	return request_coords, request_instances

def add_default_options(instances):
	for instance in instances:
		instance["requests"].append(Request(
			request_type=Request_Type.CHARGE,
			id=-1,
			pickup=None,
			dropoff=None,
			arrival_time=None,
			time_window=None,
			#max_ride_time=max_ride_time
			))
		instance["requests"].append(Request(
				request_type=Request_Type.IDLE,
				id=-2,
				pickup=None,
				dropoff=None,
				arrival_time=None,
				time_window=None,
				#max_ride_time=max_ride_time
			))
		instance["vehicles"].append(Vehicle(
			id = -1,
			location=None,
		))