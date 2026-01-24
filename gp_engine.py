#!/usr/bin/env python3

# gp_engine.py
import random
import copy
import terminals
import utils
from parallel_eval import evaluate_population_parallel

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
	def __init__(self, travel_times, simulator, train_instances, val_instances, logger = utils.Logger()):
		self.logger = logger
		self.simulator = simulator
		self.train_instances = train_instances
		self.val_instances = val_instances
		self.travel_times = travel_times
		self.population_size = 50
		self.elite_size = 1
		self.tournament_size = 5
		self.crossover_rate = 0.8
		self.mutation_rate = 0.2
		
	def train(self, n_generations=20, max_workers=None):
		population = [GPIndividual() for _ in range(self.population_size)]
		self.logger.log(f"[Init] Created {len(population)} individuals.")
		for i, ind in enumerate(population):
			self.logger.log(f"  Ind-{i}: Veh={ind.veh_policy} | Req={ind.req_policy}")
		best = None
		
		
		for gen in range(n_generations):
			print(f"Started Generation {gen}.")

			evaluate_population_parallel(
                population=population,
                simulator=self.simulator,
                travel_times=self.travel_times,
                instances=self.train_instances,
                max_workers=max_workers,
                chunksize=1,  # with 50 inds, 1 is fine; can try 2-5 later
            )

			population.sort(key=lambda x: x.fitness)
			best = population[0]
			self.logger.log(f"Gen {gen}: Best Fitness = {best.fitness:.4f}")

			
			new_pop = population[:self.elite_size]
			if gen % 5 == 0:
				for _ in range(3):  # inject fresh individuals
					new_pop.append(GPIndividual())
			while len(new_pop) < self.population_size:
				if random.random() < self.crossover_rate:
					p1 = self.tournament_select(population)
					p2 = self.tournament_select(population)
					c1, c2 = self.crossover(p1, p2)
					new_pop.extend([c1, c2])
				elif random.random() < self.mutation_rate:
					p = self.tournament_select(population)
					m = self.mutate(p)
					new_pop.append(m)
				else:
					new_pop.append(self.tournament_select(population).clone())
					
			population = new_pop[:self.population_size]
			
			fitnesses = [f for f in (ind.fitness for ind in population) if f is not None]
			if fitnesses:
				self.logger.log(f"[Gen {gen}] Best: {min(fitnesses):.2f}, Avg: {sum(fitnesses)/len(fitnesses):.2f}, Worst: {max(fitnesses):.2f}")
			else:
				self.logger.log(f"[Gen {gen}] No valid individuals this generation.")
			self.logger.log(f"         Best Veh Policy: {best.veh_policy}")
			self.logger.log(f"         Best Req Policy: {best.req_policy}")
			
			
			
		self.logger.log("\\n--- Final Best Individual ---")
		self.logger.log(f"Best Fitness: {best.fitness:.4f}")
		self.logger.log(f"Vehicle Policy (postfix): {best.veh_policy}")
		self.logger.log(f"Request Policy (postfix): {best.req_policy}")
		print("\\n--- Final Best Individual ---")
		print(f"Best Fitness: {best.fitness:.4f}")
		print(f"Vehicle Policy (postfix): {best.veh_policy}")
		print(f"Request Policy (postfix): {best.req_policy}")
		return best
	
	def evaluate(self, individual, instances, normalise=False):
		total = 0
		for inst in instances:
			score = self.simulator.run(inst, (individual.veh_policy, individual.req_policy), travel_times=self.travel_times)
			total += score
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
	