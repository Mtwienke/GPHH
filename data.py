#!/usr/bin/env python3

# data.py
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from travel_metrics import travel_metrics
from enum import Enum, auto
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
	location: Tuple[float, float]
	schedule: List[Request] = field(default_factory=list)
	capacity: int = 3
	battery_capacity: float = 15.0
	charge_level: float = 15.0
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

def generate_instances(graph, n_instances=10, n_requests=50, n_vehicles = 5, time_horizon=60*60*24, n_chargingStations = 3, seed=42, simulator=None, norming_policy = None):
	random.seed(seed)
	instances = []
	instance_seeds = random.sample(range(1, 30*n_instances), n_instances)
	nodes = [n for n, d in graph.nodes(data=True) if d.get("building")]
	for i in range(n_instances):
		requests = []
		random.seed(instance_seeds[i]*seed)
		for r in range(n_requests):
			startNode, destinationNode = random.sample(nodes, 2)
			arrival_time = random.uniform(0, time_horizon - 180)
			tstart = arrival_time
			tend = arrival_time + 30
			#max_ride_time = travel_time(u = startNode, v = destinationNode, G = graph) * 2
			requests.append(Request(
				request_type=Request_Type.RIDE,
				id=r,
				pickup=startNode,
				dropoff=destinationNode,
				arrival_time=arrival_time,
				time_window=(tstart, tend),
				#max_ride_time=max_ride_time
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


def import_requests(filepath):
    pass