# parallel_eval.py
from concurrent.futures import ProcessPoolExecutor

_WORKER_SIM = None
_WORKER_travel_times = None
_WORKER_INSTANCES = None

def _init_worker(simulator, travel_times, instances):
    global _WORKER_SIM, _WORKER_travel_times, _WORKER_INSTANCES
    # Optional: disable logging inside workers
    try:
        simulator.log = None
    except Exception:
        pass

    _WORKER_SIM = simulator
    _WORKER_travel_times = travel_times
    _WORKER_INSTANCES = instances

def _eval_individual(payload):
    idx, veh_policy, req_policy = payload
    total = 0.0
    for inst in _WORKER_INSTANCES:
        total += _WORKER_SIM.run(inst, (veh_policy, req_policy), travel_times=_WORKER_travel_times)
    return idx, total / len(_WORKER_INSTANCES)

def evaluate_population_parallel(population, simulator, travel_times, instances, max_workers=None, chunksize=1):
    tasks = [(i, ind.veh_policy, ind.req_policy)
             for i, ind in enumerate(population)
             if ind.fitness is None]
    if not tasks:
        return

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(simulator, travel_times, instances),
    ) as ex:
        for idx, fitness in ex.map(_eval_individual, tasks, chunksize=chunksize):
            population[idx].fitness = fitness