#!/usr/bin/env python3
# main.py

import utils
from gp_engine import GPEngine
from simulation import Simulator
from data import generate_instances
import random
import numpy as np
from GraphManager import GraphManager
from datetime import datetime
from terminals import generate_norming_policy

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	
	
def main():
	set_seed(42)
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	logger = utils.Logger(filename=f"main from {timestamp}.txt")
	manager = GraphManager("Detmold, Germany", include_charging=True, max_buildings= 50000, logger=logger)
	manager.loadGraph()
	G = manager.graph
	sim = Simulator(log=None, graph=G)
	#G = load_graph(place="Detmold, Germany", add_charging=True, max_buildings= 50000)
	norming_poliy = generate_norming_policy ()
	# Generate training/validation/test instances
	train_instances = generate_instances(graph = G, n_instances=5, n_requests=100, n_vehicles=10, simulator=sim, norming_policy=norming_poliy)
	val_instances = generate_instances(graph = G, n_instances=5, n_requests=100, n_vehicles=10, simulator=sim, norming_policy=norming_poliy)
	test_instances = generate_instances(graph = G, n_instances=5, n_requests=100, n_vehicles=10, simulator=sim, norming_policy=norming_poliy)
	
	# Initialize and train the GP engine

	gp = GPEngine(simulator=sim, train_instances=train_instances, val_instances=val_instances, logger=logger, graph=G)
	best_individual = gp.train(n_generations=10)
	
	logger.log("\n--- Testing Best Individual ---")
	avg_test_score = gp.evaluate(best_individual, test_instances, normalise=True)
	logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
	
	
if __name__ == '__main__':
	main()
	#G = load_graph(place="Höxter, Germany", add_charging=True, max_buildings= 50000)
	"""logger = utils.Logger()
	manager = GraphManager("Detmold, Germany", include_charging=True, max_buildings= 50000, logger=logger)
	manager.loadGraph()
	G = manager.graph
	manager.show_graph()"""