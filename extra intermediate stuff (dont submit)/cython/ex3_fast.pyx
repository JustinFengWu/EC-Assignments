# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Exercise 3 – Fast Cython Implementation
---------------------------------------
Cython-compiled inner loops for Single-Objective (SOE) and Multi-Objective (MOE)
Evolutionary Algorithms, compatible with IOH Analyzer logging.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sin
from ioh import get_problem, ProblemClass, logger

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def _rand_double():
    return rand() / <double>RAND_MAX

cpdef int cost_uniform(np.ndarray[np.uint8_t, ndim=1] x):
    """Uniform constraint cost = number of 1-bits."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef int total = 0
    for i in range(n):
        total += x[i]
    return total

cpdef np.ndarray[np.uint8_t, ndim=1] mutate_bitflip_at_least_one(
        np.ndarray[np.uint8_t, ndim=1] x, double p):
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
# SOE (Single-Objective Evolutionary Algorithm)
# ------------------------------------------------------------
def soe_uniform(problem, int seed, int pop_size, int budget):
    """Run SOE with a pre-created problem that has logger attached."""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n

    cdef np.ndarray[np.uint8_t, ndim=2] pop
    cdef np.ndarray[np.float64_t, ndim=1] fitness
    cdef np.ndarray[np.uint8_t, ndim=1] parent, child
    cdef int i, i1, i2, j, evals
    cdef double f_child

    pop = np.random.randint(0, 2, (pop_size, n), dtype=np.uint8)
    fitness = np.zeros(pop_size, dtype=np.float64)

    for i in range(pop_size):
        pop[i] = np.where(pop[i] > 0, 1, 0)
        fitness[i] = problem(pop[i].tolist())

    evals = pop_size

    while evals < budget:
        i1, i2 = np.random.randint(0, pop_size, 2)
        if fitness[i1] > fitness[i2]:
            parent = pop[i1]
        else:
            parent = pop[i2]

        child = mutate_bitflip_at_least_one(parent, p)
        f_child = problem(child.tolist())
        evals += 1

        j = int(np.argmin(fitness))
        if f_child > fitness[j]:
            pop[j] = child
            fitness[j] = f_child

    print(f"SOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")


# ------------------------------------------------------------
# MOE (Multi-Objective Evolutionary Algorithm)
# ------------------------------------------------------------
cdef inline bint dominates(double fa, int ca, double fb, int cb):
    return (fa >= fb and ca <= cb) and (fa > fb or ca < cb)

def moe_uniform(problem, int seed, int pop_size, int budget):
    """Run MOE with a pre-created problem that has logger attached."""
    srand(seed)
    np.random.seed(seed)
    
    cdef int n = problem.meta_data.n_variables
    cdef double p = 1.0 / n

    cdef np.ndarray[np.uint8_t, ndim=2] pop = np.random.randint(0, 2, (pop_size, n), dtype=np.uint8)
    cdef np.ndarray[np.float64_t, ndim=1] fitness = np.zeros(pop_size, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] cost = np.zeros(pop_size, dtype=np.int32)

    cdef int i, evals, idx, worst
    cdef np.ndarray[np.uint8_t, ndim=1] child
    cdef double f_child
    cdef int c_child

    for i in range(pop_size):
        fitness[i] = problem(pop[i].tolist())
        cost[i] = cost_uniform(pop[i])

    evals = pop_size

    while evals < budget:
        idx = np.random.randint(0, pop_size)
        child = mutate_bitflip_at_least_one(pop[idx], p)
        f_child = problem(child.tolist())
        c_child = cost_uniform(child)
        evals += 1

        worst = int(np.argmin(fitness))
        if dominates(f_child, c_child, fitness[worst], cost[worst]):
            pop[worst] = child
            fitness[worst] = f_child
            cost[worst] = c_child

    print(f"MOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")


# ------------------------------------------------------------
# Combined runner (runs both SOE and MOE, with IOH logging)
# ------------------------------------------------------------
def run_ex3():
    cdef list PROBLEMS = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]
    cdef list POP_SIZES = [10, 20, 50]
    cdef int RUNS = 30
    cdef int BUDGET = 10000

    print("=== Running SOE (Single-Objective) ===")
    
    # Create ONE logger for all SOE runs
    l1 = logger.Analyzer(
        root="ex3_fast_logs",
        folder_name="SOE",
        algorithm_name="SOE",
        algorithm_info="Exercise 3 SOE Cython"
    )

    for fid in PROBLEMS:
        print(f"Starting SOE for problem {fid}...")
        
        # Create problem and attach logger ONCE per problem
        problem = get_problem(fid=fid, instance=1, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l1)
        
        for pop_size in POP_SIZES:
            for seed in range(1, RUNS + 1):
                soe_uniform(problem, seed, pop_size, BUDGET)
                problem.reset()  # Reset for next run
        
        # Important: detach logger after all runs for this problem
        problem.detach_logger()
        print(f"✓ Completed SOE runs for problem {fid}")
    
    del l1
    print("✓ All SOE experiments finished!\n")

    print("=== Running MOE (Multi-Objective) ===")
    
    # Create ONE logger for all MOE runs
    l2 = logger.Analyzer(
        root="ex3_fast_logs",
        folder_name="MOE",
        algorithm_name="MOE",
        algorithm_info="Exercise 3 MOE Cython"
    )

    for fid in PROBLEMS:
        print(f"Starting MOE for problem {fid}...")
        
        # Create problem and attach logger ONCE per problem
        problem = get_problem(fid=fid, instance=1, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l2)
        
        for pop_size in POP_SIZES:
            for seed in range(1, RUNS + 1):
                moe_uniform(problem, seed, pop_size, BUDGET)
                problem.reset()  # Reset for next run
        
        # Important: detach logger after all runs for this problem
        problem.detach_logger()
        print(f"✓ Completed MOE runs for problem {fid}")
    
    del l2
    print("✓ All MOE experiments finished!")
