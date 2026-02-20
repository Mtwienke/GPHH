#!/usr/bin/env python3
# main.py

import utils
from gp_engine import GPEngine
from simulation import Simulator
from data import generate_instances
import data
import random
import numpy as np
from GraphManager import GraphManager
from datetime import datetime
from terminals import generate_norming_policy
from travel_metrics import travel_metrics

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	
	
def main(city = "Detmold"):
	place = city + ", Germany"
	set_seed(42)
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	logger = utils.Logger(filename=f"main from {timestamp}.txt")
	request_coords, ext_req_instances = data.import_ext_instances(place=place)
	manager = GraphManager(place=place, include_charging=True, max_buildings= 500000, logger=None, graph_folder="graphs", request_coords=request_coords)
	manager.loadGraph()
	G = manager.graph
	#G = load_graph(place="Detmold, Germany", add_charging=True, max_buildings= 50000)
	norming_policy = generate_norming_policy ()
	# Generate training/validation/test instances
	print("extracting requests from JSON")
	#external_instances = data.import_instances(manager, ext_req_instances)
	print("generating instances")
	n_requests = 20
	n_vehicles = 2
	train_instances = generate_instances(graph = G, n_instances=250, n_requests=n_requests, n_vehicles=n_vehicles, seed=42)
	val_instances = generate_instances(graph = G, n_instances=500, n_requests=n_requests, n_vehicles=n_vehicles, seed=43)
	test_instances = generate_instances(graph = G, n_instances=500, n_requests=n_requests, n_vehicles=n_vehicles, seed=44)
	print("generating a cache for travel_times")
	# Initialize and train the GP engine
	instances = train_instances + val_instances + test_instances #+ external_instances
	
	tm = travel_metrics(G, city=city)
	tm.build_cache()
	print(f"cache created")
 
	sim = Simulator(log=None, travel_metrics=tm)
	print("Simulator created")
	for set in train_instances, val_instances, test_instances:#, external_instances:
		for instance in set:
			norm_score=sim.run(instance=instance, policy_pair=norming_policy,travel_metrics=tm)
			instance['normscore']=norm_score[0]
	print ("normscores calculated")
	gp = GPEngine(simulator=sim, train_instances=train_instances, val_instances=val_instances, city=city, test_instances=test_instances, logger=None, pop_size = 1000, travel_metrics=tm)#, external_instances=external_instances)
	best_individual = gp.train(n_generations=50, max_workers=8, split_training=True, n_vehicles=n_vehicles, n_requests=n_requests)
	
	logger.log("\n--- Testing Best Individual ---")
	avg_test_score = gp.evaluate(best_individual, test_instances, normalise=True)
	logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
	print(f"test score: {avg_test_score}")
	#logger.log("\n--- Testing Best Individual on real instances ---")
	#avg_test_score = gp.evaluate(best_individual, normalise=True, external_instances, normalise=True)
	#logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
	#print(f"test score: {avg_test_score}")
	manager.show_graph()
	
	
if __name__ == '__main__':
	main(city="Höxter")
	#G = load_graph(place="Höxter, Germany", add_charging=True, max_buildings= 50000)
	#logger = utils.Logger()
	"""json = data.load_external_json("/Users/markuswienkemeier/Coding/GPHH/external_data/hx_core_avg_max/hx_core_0_avg.json")
	request_coords, requests, timedate = data.import_requests_from_external_json(json)
	manager = GraphManager("Höxter, Germany", include_charging=True, max_buildings= 500000, request_coords=request_coords)

	manager.loadGraph()

	#manager.add_request_nodes(request_coords)
	G = manager.graph
	manager.show_graph()"""