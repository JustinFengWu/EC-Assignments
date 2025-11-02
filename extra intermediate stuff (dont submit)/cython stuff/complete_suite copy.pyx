# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Complete Algorithm Suite - Cython Implementation
------------------------------------------------
RLS, (1+1)EA, GA, SOE, MOE, GSEMO with IOH logging
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sin
from ioh import get_problem, ProblemClass, logger

# ------------------------------------------------------------
# Global parameters
# ------------------------------------------------------------
BUDGET = 100000
RUNS = 30
PROBLEMS = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
cdef inline double _rand_double():
    return rand() / <double>RAND_MAX

def cost_uniform(np.ndarray[np.uint8_t, ndim=1] x):
    """Uniform constraint cost = number of 1-bits."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef int total = 0
    for i in range(n):
        total += x[i]
    return total

def mutate_bitflip_at_least_one(np.ndarray[np.uint8_t, ndim=1] x, double p):
    """Bit-flip mutation with prob p; ensure at least one flip."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] y = x.copy()
    cdef bint flipped = False
    for i in range(n):
        if _rand_double() < p:
            y[i] ^= 1
            flipped = True
    if not flipped:
        i = int(_rand_double() * n)
        y[i] ^= 1
    return y


# ------------------------------------------------------------
# RLS (Randomised Local Search)
# ------------------------------------------------------------
def rls(problem, int seed, int budget):
    """Flips one random bit per iteration"""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef int i, eval_count
    cdef double current_fitness, new_fitness
    
    current_solution = np.random.randint(0, 2, n, dtype=np.uint8)
    current_fitness = problem(current_solution.tolist())
    eval_count = 1
    
    while eval_count < budget:
        new_solution = current_solution.copy()
        i = np.random.randint(n)
        new_solution[i] ^= 1
        new_fitness = problem(new_solution.tolist())
        eval_count += 1
        
        if new_fitness >= current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    
    print(f"RLS done | fid={problem.meta_data.problem_id} | seed={seed}")


# ------------------------------------------------------------
# (1+1) EA
# ------------------------------------------------------------
def one_plus_one_ea(problem, int seed, int budget):
    """Flips each bit independently with probability 1/n"""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n
    cdef int eval_count
    cdef double current_fitness, new_fitness
    
    current_solution = np.random.randint(0, 2, n, dtype=np.uint8)
    current_fitness = problem(current_solution.tolist())
    eval_count = 1
    
    while eval_count < budget:
        new_solution = current_solution.copy()
        flips = np.random.rand(n) < p
        if flips.any():
            new_solution[flips] ^= 1
        new_fitness = problem(new_solution.tolist())
        eval_count += 1
        
        if new_fitness >= current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    
    print(f"(1+1)EA done | fid={problem.meta_data.problem_id} | seed={seed}")


# ------------------------------------------------------------
# Genetic Algorithm
# ------------------------------------------------------------
def genetic_algorithm(problem, int seed, int budget):
    """GA with tournament selection, uniform crossover, mutation"""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef int mu = 20
    cdef int lambda_ = 40
    cdef double pc = 0.9
    cdef double pm = 1.0 / n
    cdef int k = 3
    cdef int eval_count = 0
    cdef int i, remaining
    
    # Initialize population
    pop = np.random.randint(0, 2, (mu, n), dtype=np.uint8)
    fit = np.array([problem(pop[i].tolist()) for i in range(mu)], dtype=np.float64)
    eval_count = mu
    
    while eval_count < budget:
        offspring = []
        
        # Generate lambda offspring
        while len(offspring) < lambda_:
            # Tournament selection
            idx1 = np.random.randint(0, mu, k)
            idx2 = np.random.randint(0, mu, k)
            p1 = pop[idx1[np.argmax(fit[idx1])]].copy()
            p2 = pop[idx2[np.argmax(fit[idx2])]].copy()
            
            # Crossover
            if np.random.rand() < pc:
                mask = np.random.rand(n) < 0.5
                c1 = np.where(mask, p1, p2).astype(np.uint8)
                c2 = np.where(mask, p2, p1).astype(np.uint8)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Mutation
            mask1 = np.random.rand(n) < pm
            c1[mask1] = 1 - c1[mask1]
            offspring.append(c1)
            
            if len(offspring) < lambda_:
                mask2 = np.random.rand(n) < pm
                c2[mask2] = 1 - c2[mask2]
                offspring.append(c2)
        
        # Evaluate offspring
        offspring = np.array(offspring[:lambda_])
        off_fit = np.array([problem(offspring[i].tolist()) for i in range(lambda_)])
        eval_count += lambda_
        
        # (mu+lambda) selection
        combined = np.vstack([pop, offspring])
        combined_fit = np.concatenate([fit, off_fit])
        idx = np.argsort(-combined_fit)[:mu]
        pop, fit = combined[idx], combined_fit[idx]
        
        if eval_count >= budget:
            break
    
    print(f"GA done | fid={problem.meta_data.problem_id} | seed={seed}")


# ------------------------------------------------------------
# SOE (Single-Objective Evolutionary Algorithm)
# ------------------------------------------------------------
def soe_uniform(problem, int seed, int pop_size, int budget):
    """SOE with uniform cost"""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n
    cdef int i, i1, i2, j, eval_count
    cdef double f_child
    
    pop = np.random.randint(0, 2, (pop_size, n), dtype=np.uint8)
    fitness = np.zeros(pop_size, dtype=np.float64)
    
    for i in range(pop_size):
        fitness[i] = problem(pop[i].tolist())
    
    eval_count = pop_size
    
    while eval_count < budget:
        i1, i2 = np.random.randint(0, pop_size, 2)
        if fitness[i1] > fitness[i2]:
            parent = pop[i1].copy()
        else:
            parent = pop[i2].copy()
        
        # Mutate and evaluate
        child_mutated = mutate_bitflip_at_least_one(parent, p)
        f_child = problem(child_mutated.tolist())
        eval_count += 1
        
        # Replace worst if better
        j = int(np.argmin(fitness))
        if f_child > fitness[j]:
            pop[j] = np.asarray(child_mutated, dtype=np.uint8)
            fitness[j] = f_child
    
    print(f"SOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")


# ------------------------------------------------------------
# MOE (Multi-Objective Evolutionary Algorithm)
# ------------------------------------------------------------
cdef inline bint dominates(double fa, int ca, double fb, int cb):
    return (fa >= fb and ca <= cb) and (fa > fb or ca < cb)

def moe_uniform(problem, int seed, int pop_size, int budget):
    """MOE with Pareto dominance"""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n
    cdef int i, eval_count, idx, worst
    cdef double f_child
    cdef int c_child
    
    pop = np.random.randint(0, 2, (pop_size, n), dtype=np.uint8)
    fitness = np.zeros(pop_size, dtype=np.float64)
    cost = np.zeros(pop_size, dtype=np.int32)
    
    for i in range(pop_size):
        fitness[i] = problem(pop[i].tolist())
        cost[i] = cost_uniform(pop[i])
    
    eval_count = pop_size
    
    while eval_count < budget:
        idx = np.random.randint(0, pop_size)
        parent = pop[idx].copy()
        
        # Mutate and evaluate
        child_mutated = mutate_bitflip_at_least_one(parent, p)
        f_child = problem(child_mutated.tolist())
        c_child = cost_uniform(child_mutated)
        eval_count += 1
        
        # Replace worst if dominates
        worst = int(np.argmin(fitness))
        if dominates(f_child, c_child, fitness[worst], cost[worst]):
            pop[worst] = np.asarray(child_mutated, dtype=np.uint8)
            fitness[worst] = f_child
            cost[worst] = c_child
    
    print(f"MOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")


# ------------------------------------------------------------
# GSEMO (Global Simple Evolutionary Multi-Objective)
# ------------------------------------------------------------
def insert_pareto(population, candidate):
    """Insert candidate into Pareto archive"""
    child_sol, child_fit, child_cost = candidate
    
    # Check if dominated
    for sol, fit, cost in population:
        if (fit >= child_fit and cost <= child_cost) and (fit > child_fit or cost < child_cost):
            return False
    
    # Remove dominated solutions
    new_pop = []
    for sol, fit, cost in population:
        if not ((child_fit >= fit and child_cost <= cost) and (child_fit > fit or child_cost < cost)):
            new_pop.append((sol, fit, cost))
    
    new_pop.append(candidate)
    population[:] = new_pop
    return True

def gsemo(problem, int seed, int budget):
    """GSEMO with Pareto archive"""
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n
    cdef int eval_count = 0
    
    # Initialize with all-zeros
    init_sol = np.zeros(n, dtype=np.uint8)
    init_fit = problem(init_sol.tolist())
    init_cost = int(np.sum(init_sol))
    
    population = [(init_sol.copy(), init_fit, init_cost)]
    eval_count = 1
    
    while eval_count < budget:
        # Select random parent
        parent_sol, parent_fit, parent_cost = population[np.random.randint(len(population))]
        
        # Mutate
        child_sol = mutate_bitflip_at_least_one(parent_sol, p)
        child_fit = problem(child_sol.tolist())
        child_cost = int(np.sum(child_sol))
        eval_count += 1
        
        # Try to insert into archive
        insert_pareto(population, (child_sol.copy(), child_fit, child_cost))
    
    print(f"GSEMO done | fid={problem.meta_data.problem_id} | seed={seed}")


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def run_all_algorithms():
    """Run all algorithms on all problems"""
    
    algorithms = [
        ("MOE", lambda p, s: moe_uniform(p, s, 50, BUDGET)),
        ("GSEMO", lambda p, s: gsemo(p, s, BUDGET))
    ]
    
    for algo_name, algo_func in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name}...")
        print(f"{'='*60}")
        
        # Create logger for this algorithm
        l = logger.Analyzer(
            root="complete_logs",
            folder_name=algo_name,
            algorithm_name=algo_name,
            algorithm_info=f"Complete Suite - {algo_name}"
        )
        
        for fid in PROBLEMS:
            print(f"\nProblem {fid}:")
            
            problem = get_problem(fid=fid, instance=1, problem_class=ProblemClass.GRAPH)
            problem.attach_logger(l)
            
            for run in range(1, RUNS + 1):
                print(f"  Run {run}/{RUNS}", end='\r')
                algo_func(problem, run)
                problem.reset()
            
            problem.detach_logger()
            print(f"  ✓ Completed {RUNS} runs for problem {fid}")
        
        del l
        print(f"✓ {algo_name} complete!\n")
    
    print("\n" + "="*60)
    print("ALL ALGORITHMS FINISHED!")
    print("="*60)
