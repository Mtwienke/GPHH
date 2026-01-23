#!/usr/bin/env python3

# terminals.py
import operator
import math
from data import travel_time, energy_cost, euclidean_distance
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


# Terminal feature functions
TERMINALS = {
	'TVPU': lambda v, r, graph, **kwargs: travel_time(v.location, r.pickup, graph),
	'COST': lambda v, r, graph, **kwargs: travel_time(v.location, r.pickup, graph) + travel_time(r.pickup, r.dropoff, graph),
	'DEM': lambda v, r, **kwargs: r.demand,
	'DUR': lambda v, r, graph, **kwargs: travel_time(r.pickup, r.dropoff, graph),
	'SLACK': lambda v, r, graph, t=None, **kwargs: max(r.time_window[1] - t - travel_time(v.location, r.pickup, graph), 0),
	'RQ': lambda v, r, **kwargs: v.capacity,
	'RT': lambda v, r, **kwargs: v.charge_level,
	'FRT': lambda v, r, **kwargs: v.charge_level,
	'TVC': lambda v, r, graph, stations=None, **kwargs: min([travel_time(v.location, s.location, graph) for s in stations]),
}


# A very basic tree evaluator (postfix format)
def evaluate_policy(graph, policy_tree, vehicle, request, requests, vehicles, stations, current_time, logger = utils.Logger()):
	stack = []
	logger.log(f"\n--- Evaluating Policy ---")
	logger.log(f"Policy Tree (postfix): {policy_tree}")
	for token in policy_tree:
		logger.log(f"Token: {token}")
		if token in FUNCTIONS:
			if len(stack) < 2:
				logger.log("  ❌ Not enough operands on stack for function.")
				logger.log(f"  Stack before error: {stack}")
				return float('inf')
			b = stack.pop()
			a = stack.pop()
			result = FUNCTIONS[token](a, b)
			logger.log(f"  Apply {token} to {a}, {b} => {result}")
			stack.append(result)
		elif token in TERMINALS:
			try:
				val = TERMINALS[token](vehicle, request, requests=requests, vehicles=vehicles, stations=stations, t=current_time, graph=graph)
				logger.log(f"  Terminal {token} => {val}")
				stack.append(val)
			except Exception as e:
				logger.log(f"  ❌ Error evaluating terminal {token}: {traceback.format_exc()}")
				return float('inf')
		else:
			try:
				const_val = float(token)
				logger.log(f"  Constant {token} => {const_val}")
				stack.append(const_val)
			except ValueError:
				logger.log(f"  ❌ Invalid constant: {token}")
				return float('inf')
		logger.log(f"  Stack now: {stack}")
		
	if stack:
		logger.log(f"✅ Final Result: {stack[0]}\n")
		return stack[0]
	else:
		logger.log("❌ Empty stack at end of evaluation.")
		return float('inf')
	
	