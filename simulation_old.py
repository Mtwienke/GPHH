#!/usr/bin/env python3

# simulation.py
import heapq
from terminals import evaluate_policy
from data import travel_time, energy_cost, Vehicle, Request
from copy import deepcopy
import utils

class State:
	def __init__(self, time, eventQueue, fleet, graph):
		self.currentTime = time
		self.eventQueue = eventQueue
		self.fleet = fleet
		self.graph = graph


class Event:
	def __init__(self, time, type, data):
		self.time = time
		self.type = type  # 'request_arrival', 'vehicle_empty', 'vehicle_recharged'
		self.data = data  # request or vehicle object
		
	def __lt__(self, other):
		return self.time < other.time
	
	
class Simulator:
	def __init__(self, phi=0.01, beta=0.0005, alpha=0.05, penalty_weight=2.0, logger = utils.Logger()):
		self.logger = logger
		self.phi = phi
		self.beta = beta
		self.alpha = alpha
		self.penalty_weight = penalty_weight
		
	def run(self, instance, policy_pair):
		self.requests = {r.id: r for r in instance['requests']}
		self.vehicles = {v.id: deepcopy(v) for v in instance['vehicles']}
		self.stations = instance['stations']
		self.time = 0
		self.waiting_pool = []
		self.routes = {v.id: [] for v in self.vehicles.values()}
		
		self.event_queue = []
		for r in self.requests.values():
			heapq.heappush(self.event_queue, Event(r.arrival_time, 'request_arrival', r))
			
		while self.event_queue:
			event = heapq.heappop(self.event_queue)
			self.time = event.time
			if event.type == 'request_arrival':
				self.handle_request_arrival(event.data, policy_pair[0])
			for vehicle in self.vehicles.values():
				if vehicle.schedule == [] and vehicle.onboard == []:
					self.handle_idle_vehicle(vehicle, policy_pair[1])  # π_req
				
		return self.compute_objective()
	
	def handle_request_arrival(self, request, vehicle_policy):
		candidate_vehicles = []
		for vehicle in self.vehicles.values():
			priority = evaluate_policy(vehicle_policy, vehicle, request, self.requests, self.vehicles, self.stations, self.time)
			candidate_vehicles.append((priority, vehicle))
			
		candidate_vehicles.sort(key=lambda x: x[0])
		best_vehicle = candidate_vehicles[0][1] if candidate_vehicles else None
		
		if best_vehicle:
			self.assign_request(best_vehicle, request)
		else:
			self.waiting_pool.append(request)
	
	def handle_idle_vehicle(self, vehicle, request_policy):
		# Skip if no available requests
		pending = [r for r in self.requests.values() if not r.assigned and self.time >= r.earliest]
		if not pending:
			return
		
		best_score = float('inf')
		best_request = None
		for req in pending:
			score = evaluate_policy(request_policy, vehicle, req, self.requests, self.vehicles, self.stations, self.time)
			if score < best_score:
				best_score = score
				best_request = req
				
		if best_request:
			self.assign_request(vehicle, best_request)
			best_request.assigned = True
			self.logger and self.logger.log(f"[π_req] Vehicle {vehicle.id} assigned to Request {best_request.id} at t={self.time}")
			
	def assign_request(self, vehicle, request):
		travel = travel_time(vehicle.location, request.pickup)
		cost = travel + travel_time(request.pickup, request.dropoff)
		self.routes[vehicle.id].append((self.time + travel, request))
		vehicle.schedule.append(request)
		vehicle.location = request.dropoff
		vehicle.charge_level -= energy_cost(vehicle.location, request.pickup) + energy_cost(request.pickup, request.dropoff)
		
	def compute_objective(self):
		total_travel = 0
		late_penalty = 0
		for route in self.routes.values():
			for start_time, request in route:
				total_travel += travel_time(request.pickup, request.dropoff)
				if start_time > request.time_window[1]:
					late_penalty += (start_time - request.time_window[1])
					
		return total_travel + self.penalty_weight * late_penalty
	