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
import os
import pandas as pd
from pathlib import Path

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	
	
def main():
	set_seed(42)
	cities = "Paderborn", "Höxter"
	Ns_requests = 20, 50, 200
	Ns_vehicles = 2, 5, 20
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	logger = utils.Logger(filename=f"main from {timestamp}.txt")
	policies = []
	testsets_list = []
	simulators = {}

	for city in cities:
		print(f"--------------- starting run for {city} ---------------")
		for r in Ns_requests:
			print(f"--------------- starting run for {city} with n_requests = {r} ---------------")
			for v in Ns_vehicles:
				print(f"--------------- starting run for {city} with n_requests = {r} and n_vehicles = {v} ---------------")
				best, test_instances, sim = call_run(city=city, n_requests=r, n_vehicles=v, logger=logger)

				policies.append({
					"policy_city": city,
					"policy_r": r,
					"policy_v": v,
					"veh_policy": best.veh_policy,
					"req_policy": best.req_policy,
					# optional:
					# "policy_id": best.uid or hash(...)
				})

				testsets_list.append({
					"test_city": city,
					"test_r": r,
					"test_v": v,
					"instances": test_instances,
				})
		simulators[city] = sim
	PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
	RESULT_DIR = PROJECT_ROOT / "results"
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	dir_name = f"cross_eval_{timestamp}"
	base_dir = RESULT_DIR / dir_name
	out_path = base_dir / "cross_eval.parquet"
	cross_eval_all_vs_all(sim_by_city=simulators, policies=policies, testsets_list=testsets_list, out_path=out_path)

def cross_eval_all_vs_all(sim_by_city, policies, testsets_list, out_path: Path):
    rows = []

    for p in policies:
        for ts in testsets_list:
            sim = sim_by_city[ts["test_city"]]   # choose simulator by INSTANCE city

            for k, inst in enumerate(ts["instances"]):
                inst_id = inst.get("id", k)

                score_total, score_norm = sim.run(
                    inst,
                    (p["veh_policy"], p["req_policy"]),
                    # no travel_metrics arg needed if sim already owns it
                )

                rows.append({
                    "phase": "cross_eval",
                    "run_type": "all_vs_all",
                    "policy_city": p["policy_city"],
                    "policy_r": p["policy_r"],
                    "policy_v": p["policy_v"],
                    "test_city": ts["test_city"],
                    "test_r": ts["test_r"],
                    "test_v": ts["test_v"],
                    "instance_id": inst_id,
                    "score_total": float(score_total),
                    "score_norm": float(score_norm),
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    
def call_run(city, n_requests, n_vehicles, logger):
	place = city + ", Germany"

	request_coords, ext_req_instances = data.import_ext_instances(place=place)
	manager = GraphManager(place=place, include_charging=True, max_buildings= 500000, logger=None, graph_folder="graphs", request_coords=request_coords)
	manager.loadGraph()
	G = manager.graph
	norming_policy = generate_norming_policy ()
	# Generate training/validation/test instances
	print("extracting requests from JSON")
	external_instances = data.import_instances(manager, ext_req_instances)
	print("generating instances")
	train_instances = generate_instances(graph = G, n_instances=250, n_requests=n_requests, n_vehicles=n_vehicles)
	val_instances = generate_instances(graph = G, n_instances=500, n_requests=n_requests, n_vehicles=n_vehicles)
	test_instances = generate_instances(graph = G, n_instances=500, n_requests=n_requests, n_vehicles=n_vehicles)
	print("generating a cache for travel_times")
	# Initialize and train the GP engine
	instances = train_instances + val_instances + test_instances + external_instances
	
	tm = travel_metrics(G, city=city)
	tm.build_cache()
	print(f"cache created")
 
	sim = Simulator(log=None, travel_metrics=tm)
	print("Simulator created")
	for set in train_instances, val_instances, test_instances, external_instances:
		for instance in set:
			norm_score=sim.run(instance=instance, policy_pair=norming_policy,travel_metrics=tm)
			instance['normscore']=norm_score[0]
	print ("normscores calculated")
	data.add_default_options(instances=instances)
	gp = GPEngine(simulator=sim, train_instances=train_instances, val_instances=val_instances, city=city, test_instances=test_instances, logger=None, pop_size = 1000, travel_metrics=tm, external_instances=external_instances)
	best_individual = gp.train(n_generations=50, max_workers=8, split_training=True, n_vehicles=n_vehicles, n_requests=n_requests)
	return best_individual, test_instances, sim
	#logger.log("\n--- Testing Best Individual ---")
	#avg_test_score = gp.evaluate(best_individual, test_instances, normalise=True)
	#logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
	#print(f"test score: {avg_test_score}")
	#logger.log("\n--- Testing Best Individual on real instances ---")
	#avg_test_score = gp.evaluate(best_individual, normalise=True, external_instances, normalise=True)
	#logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
	#print(f"test score: {avg_test_score}")
	#manager.show_graph()


	
if __name__ == '__main__':
	#main()
	#G = load_graph(place="Höxter, Germany", add_charging=True, max_buildings= 50000)
	#logger = utils.Logger()
	"""json = data.load_external_json("/Users/markuswienkemeier/Coding/GPHH/external_data/pb_nachfragen/pb_2738_0_0.json")
	request_coords, requests = data.import_ext_instances(place="Paderborn, Germany")"""
	manager = GraphManager("Höxter, Germany", include_charging=True, max_buildings= 500000)#request_coords=request_coords, geojson_path="external_data/sicpArea_merged_final.geojson")

	manager.loadGraph()
	PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
	graphdir = PROJECT_ROOT / "graphs"
	outpath = graphdir / "Höxter_with_requestnodes.png"
	#manager.add_request_nodes(request_coords)
	G = manager.graph
	manager.show_graph()#"""