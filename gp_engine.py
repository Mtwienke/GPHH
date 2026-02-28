#!/usr/bin/env python3

# gp_engine.py
import random
import copy
import terminals
import utils
from parallel_eval import evaluate_population_parallel
from pathlib import Path

FUNCTION_SET = ['+', '-', '*', '/', 'min', 'max']
TERMINAL_SET = ['TVPU', 'COST', 'DEM', 'DUR', 'SLACK', 'RQ', 'RT', 'FRT', 'TVC']


class GPIndividual:
	def __init__(self, veh_policy=None, req_policy=None):
		self.veh_policy = veh_policy or terminals.random_tree(4)
		self.req_policy = req_policy or terminals.random_tree(4)
		self.fitness = None
		
	@staticmethod
	def random_tree(max_depth=1, current_depth=0):
		if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
			return [random.choice(TERMINAL_SET)]
		else:
			left = GPIndividual.random_tree(max_depth, current_depth + 1)
			right = GPIndividual.random_tree(max_depth, current_depth + 1)
			op = random.choice(FUNCTION_SET)
			return left + right + [op]
		
	def clone(self):
		return copy.deepcopy(self)
	
	
class GPEngine:
	def __init__(self, travel_metrics, simulator, train_instances, val_instances, city, test_instances = None, external_instances = None, logger = utils.Logger(), pop_size = 1000):
		self.logger = utils.Logger()
		self.simulator = simulator
		self.train_instances = train_instances
		self.val_instances = val_instances
		self.travel_metrics = travel_metrics
		self.population_size = pop_size
		self.elite_size = 0
		self.tournament_size = 7
		self.crossover_rate = 0.8
		self.mutation_rate = 0.15
		self.reproduction_rate = 0.05
		self.city = city
		self.external_instances = external_instances
		self.test_instances = test_instances
		
	def train(self, n_vehicles, n_requests, n_generations=20, max_workers=2, split_training = False):
		PROJECT_ROOT = Path(__file__).resolve().parent  # oder parent.parent wenn nötig
		RESULT_DIR = PROJECT_ROOT / "results"
		dir_name = f"{self.city}_v{n_vehicles}_r{n_requests}"
		base_dir = RESULT_DIR / dir_name
		population = [GPIndividual() for _ in range(self.population_size)]
		self.logger.log(f"[Init] Created {len(population)} individuals.")
		for i, ind in enumerate(population):
			self.logger.log(f"  Ind-{i}: Veh={ind.veh_policy} | Req={ind.req_policy}")
		best = None
		
		if split_training:
			instances_per_gen = len(self.train_instances)//n_generations
		best_per_gen = list()
		
		for gen in range(n_generations):
			print(f"Started Generation {gen}.")
			if split_training:
				try:
					instances = self.train_instances[gen*instances_per_gen:(gen+1)*instances_per_gen]
				except:
					instances = self.train_instances
			else:
				instances = self.train_instances
			evaluate_population_parallel(
                population=population,
                simulator=self.simulator,
                travel_metrics=self.travel_metrics,
                instances=instances,
                base_dir=base_dir,
                phase="train",
                gen=gen,
                max_workers=max_workers,
                chunk_size=50,  # with 50 inds, 1 is fine; can try 2-5 later
                force=True
            )

			population.sort(key=lambda x: x.fitness)
			best = population[0]
			if best_per_gen:
				best_per_gen.append(copy.deepcopy(best))
				print("Bests per gen: ")
				for gen in range(len(best_per_gen)):
					print(f"Gen {gen}: vehicle policy:  {best_per_gen[gen].veh_policy} --- request policy {best_per_gen[gen].req_policy}")
			else:
				best_per_gen = [copy.deepcopy(best)]
				print(f"in else with best_per_gen {best_per_gen}")
			self.logger.log(f"Gen {gen}: Best Fitness = {best.fitness:.4f}")

			
			new_pop = []
			"""if gen % 5 == 0:
				for _ in range(3):  # inject fresh individuals
					new_pop.append(GPIndividual())"""
			while len(new_pop) < self.population_size:
				random_number = random.random()
				if random_number < self.crossover_rate:
					p1 = self.tournament_select(population)
					p2 = self.tournament_select(population)
					c1, c2 = self.crossover(p1, p2)
					new_pop.extend([c1, c2])
				elif random_number < self.mutation_rate + self.crossover_rate:
					p = self.tournament_select(population)
					m = self.mutate(p)
					new_pop.append(m)
				elif random_number < self.mutation_rate + self.crossover_rate + self.reproduction_rate:
					new_pop.append(self.tournament_select(population).clone())
				else:
					print("something strange happened here")
					
			population = new_pop
			
		print("\\n--- Validating Best Individuals Per Generation ---")
		evaluate_population_parallel(
                population=best_per_gen,
                simulator=self.simulator,
                travel_metrics=self.travel_metrics,
                instances=self.val_instances,
                base_dir=base_dir,
                phase="validation",
                gen=None,
                max_workers=max(5, max_workers),
                chunk_size=10,  # with 50 inds, 1 is fine; can try 2-5 later
                force = True
			)

		for gen in range(len(best_per_gen)):
			self.simulator.run(self.val_instances[0], [best_per_gen[gen].veh_policy, best_per_gen[gen].req_policy], self.travel_metrics, plot_dir = f"Gen {gen}")
		best_per_gen.sort(key=lambda x: x.fitness)
		best = best_per_gen[0]
		self.logger.log(f"Validation: Best Fitness = {best.fitness:.4f}")
		print(f"Validation: Best Fitness = {best.fitness:.4f}")
  
  
		self.logger.log("\\n--- Final Best Individual ---")
		self.logger.log(f"Best Fitness: {best.fitness:.4f}")
		self.logger.log(f"Vehicle Policy (postfix): {best.veh_policy}")
		self.logger.log(f"Request Policy (postfix): {best.req_policy}")
		print("\\n--- Final Best Individual ---")
		print(f"Best Fitness at Validation: {best.fitness:.4f}")
		print(f"Vehicle Policy (postfix): {best.veh_policy}")
		print(f"Request Policy (postfix): {best.req_policy}")

		if self.test_instances:
			self.logger.log("\n--- Testing Best Individual ---")
			evaluate_population_parallel(
                population=[best],
                simulator=self.simulator,
                travel_metrics=self.travel_metrics,
                instances=self.test_instances,
                base_dir=base_dir,
                phase="test",
                gen=None,
                max_workers=1, # with 50 inds, 1 is fine; can try 2-5 later
                force = True
			)
			avg_test_score = self.evaluate(best, self.test_instances, normalise=True)
			self.logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
			print(f"test score: {avg_test_score}")
		if self.external_instances:
			self.logger.log("\n--- Testing Best Individual on real instances ---")
			evaluate_population_parallel(
                population=[best],
                simulator=self.simulator,
                travel_metrics=self.travel_metrics,
                instances=self.external_instances,
                base_dir=base_dir,
                phase="real",
                gen=None,
                max_workers=1,  # with 50 inds, 1 is fine; can try 2-5 later
                force = True
			)
			avg_test_score = self.evaluate(best, self.external_instances, normalise=True)
			self.logger.log(f"Test performance (normalised): {avg_test_score:.4f}")
			print(f"test score: {avg_test_score}")
		return best
	
	def evaluate(self, individual, instances, normalise=False):
		total = 0
		for inst in instances:
			score = self.simulator.run(inst, (individual.veh_policy, individual.req_policy), travel_metrics=self.travel_metrics)
			total += score[1]
		return total / len(instances)
	
	def tournament_select(self, population):
		group = random.sample(population, self.tournament_size)
		return min(group, key=lambda x: x.fitness)
	
	def crossover(self, parent1, parent2):
		child1, child2 = terminals.subtree_crossover_ast(parent1, parent2)
		return child1, child2
	
	def mutate(self, parent):
		mutant = parent.clone()
		if random.random() < 0.5:
			mutant.veh_policy = terminals.subtree_mutation_ast(mutant.veh_policy)
		else:
			mutant.req_policy = terminals.subtree_mutation_ast(mutant.req_policy)
		mutant.fitness = None
		return mutant
	