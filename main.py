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
from travel_metrics import travel_metrics

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
	#G = load_graph(place="Detmold, Germany", add_charging=True, max_buildings= 50000)
	norming_policy = generate_norming_policy ()
	# Generate training/validation/test instances
	print("generating instances")
	train_instances = generate_instances(graph = G, n_instances=5, n_requests=500, n_vehicles=50)
	val_instances = generate_instances(graph = G, n_instances=5, n_requests=500, n_vehicles=50)
	test_instances = generate_instances(graph = G, n_instances=5, n_requests=500, n_vehicles=50)
	
	print("generating a cache for travel_times")
	# Initialize and train the GP engine
	instances = train_instances + val_instances + test_instances
	tm = travel_metrics()
	tm.build_dense_tt_td(G, instances)
	print(f"cache created with {len(tm.node_to_idx)} nodes")
 
	sim = Simulator(log=None, travel_metrics=tm)
	for set in train_instances, val_instances, test_instances:
		for instance in set:
			norm_score=sim.run(instance=instance, policy_pair=norming_policy,travel_metrics=tm)
			instance['normscore']=norm_score
	gp = GPEngine(simulator=sim, train_instances=train_instances, val_instances=val_instances, logger=logger, travel_metrics=tm)
	best_individual = gp.train(n_generations=100)
	
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