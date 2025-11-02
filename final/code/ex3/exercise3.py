# ==============================================================
# Exercise 3 - New Approaches for Monotone Submodular Optimisation
# --------------------------------------------------------------
# Implements:
# 1. Population-based Single-Objective EA (SOE)
# 2. Population-based Multi-Objective EA (MOE)
# Shared utilities and IOHAnalyzer logging
# ==============================================================
import random
import numpy as np
from ioh import get_problem, ProblemClass, logger

# -------------------- PARAMETERS --------------------
RUNS = 30
BUDGET = 10000
POP_SIZES = [10, 20, 50]
PROBLEMS = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]  # MaxCoverage + MaxInfluence

# -------------------- SHARED HELPERS --------------------

def cost_uniform(x):
    """Uniform constraint: cost = number of chosen items."""
    return int(sum(x))

def mutate_bitflip_at_least_one(x, p):
    """Mutation: flip bits with probability p, ensure at least one flip."""
    n = len(x)
    flip_mask = np.random.rand(n) < p
    if not flip_mask.any():
        flip_mask[np.random.randint(0, n)] = True
    y = x.copy()
    for i, flip in enumerate(flip_mask):
        if flip:
            y[i] ^= 1
    return y

def repair_uniform(problem, x, B):
    """
    Repair step for constraint handling.
    If |x| > B, iteratively remove least valuable elements.
    (Matches Exercise 3 guidance allowing problem-specific repair.)
    """
    if cost_uniform(x) <= B:
        return x
    x = x.copy()
    while cost_uniform(x) > B:
        ones = [i for i, bit in enumerate(x) if bit == 1]
        if not ones:
            break
        base_val = problem(x)
        worst_i, min_loss = None, float('inf')
        for i in ones:
            x[i] = 0
            loss = base_val - problem(x)
            if loss < min_loss:
                min_loss, worst_i = loss, i
            x[i] = 1
        x[worst_i] = 0
    return x

# ---- Shared for multi-objective EA ----
def dominates(a, b):
    """True if a dominates b (maximize reward, minimize cost)."""
    (_, fa, ca) = a
    (_, fb, cb) = b
    return (fa >= fb and ca <= cb) and (fa > fb or ca < cb)

def fast_non_dominated_sort(pop):
    """Basic NSGA-II sorting."""
    S = {i: [] for i in range(len(pop))}
    n = [0]*len(pop)
    fronts = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q: continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return [f for f in fronts if f]

def crowding_distance(front, pop):
    """Measures solution spread for diversity"""
    if not front:
        return {}
    distances = {idx: 0.0 for idx in front}
    # Objective 1: reward
    front_f = sorted(front, key=lambda i: pop[i][1])
    fvals = [pop[i][1] for i in front_f]
    if max(fvals) > min(fvals):
        for k in range(1, len(front_f)-1):
            distances[front_f[k]] += (fvals[k+1]-fvals[k-1])/(max(fvals)-min(fvals))
    # Objective 2: cost
    front_c = sorted(front, key=lambda i: pop[i][2])
    cvals = [pop[i][2] for i in front_c]
    if max(cvals) > min(cvals):
        for k in range(1, len(front_c)-1):
            distances[front_c[k]] += (cvals[k-1]-cvals[k+1])/(max(cvals)-min(cvals))
    return distances

# -------------------- SINGLE-OBJECTIVE EA --------------------

def soe_uniform(problem, seed, pop_size, budget):
    """
    Population-based Single-Objective EA (Exercise 3 - Part 1)
    ----------------------------------------------------------
    - Keeps a population of candidate solutions
    - Maximizes reward f(x)
    - Repairs any solution that exceeds uniform constraint B
    - Maintains diversity by removing duplicates
    """
    random.seed(seed)
    np.random.seed(seed)
    n = problem.meta_data.n_variables
    B = max(1, int(round(0.1 * n)))
    p = 1.0 / n

    # --- Initialize population ---
    pop = []
    for _ in range(pop_size):
        x = [random.randint(0, 1) for _ in range(n)]
        x = repair_uniform(problem, x, B)
        f = problem(x)
        pop.append((x, f, cost_uniform(x)))
    evals = len(pop)

    # --- Evolutionary loop ---
    while evals < budget:
        # 1. Parent selection (binary tournament)
        a, b = random.sample(pop, 2)
        parent = a if a[1] >= b[1] else b

        # 2. Variation (mutation)
        child = mutate_bitflip_at_least_one(parent[0], p)
        child = repair_uniform(problem, child, B)
        f_child = problem(child)
        evals += 1
        pop.append((child, f_child, cost_uniform(child)))

        # 3. Diversity maintenance
        unique = {}
        for ind in pop:
            key = tuple(ind[0])
            if key not in unique or ind[1] > unique[key][1]:
                unique[key] = ind
        pop = list(unique.values())

        # 4. Survivor selection (μ+1 elitism)
        pop.sort(key=lambda t: t[1], reverse=True)
        pop = pop[:pop_size]

    print(f"SOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")

# -------------------- MULTI-OBJECTIVE EA --------------------

def moe_uniform(problem, seed, pop_size, budget):
    """
    Population-based Multi-Objective EA (Exercise 3 - Part 2)
    ----------------------------------------------------------
    - Two objectives: maximize reward f(x), minimize cost |x|
    - Uses non-dominated sorting (Pareto ranking)
    - Uses crowding distance for diversity
    """
    random.seed(seed)
    np.random.seed(seed)
    n = problem.meta_data.n_variables
    B = max(1, int(round(0.1 * n)))
    p = 1.0 / n

    # --- Initialize population ---
    pop = []
    for _ in range(pop_size):
        x = [random.randint(0, 1) for _ in range(n)]
        f = problem(x)
        c = cost_uniform(x)
        pop.append((x, f, c))
    evals = len(pop)

    # --- Evolutionary loop ---
    while evals < budget:
        parent = random.choice(pop)
        childx = mutate_bitflip_at_least_one(parent[0], p)
        f_child = problem(childx)
        c_child = cost_uniform(childx)
        evals += 1
        pop.append((childx, f_child, c_child))

        # --- NSGA-II selection ---
        fronts = fast_non_dominated_sort(pop)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                dist = crowding_distance(front, pop)
                front.sort(key=lambda i: dist.get(i, 0), reverse=True)
                new_pop.extend(front[:pop_size - len(new_pop)])
                break
        pop = [pop[i] for i in new_pop]

    print(f"MOE done | fid={problem.meta_data.problem_id} | pop={pop_size} | seed={seed}")

# -------------------- RUNNERS --------------------

def run_single_objective():
    """Runs Exercise 3 Part 1 experiments and logs to IOHAnalyzer (same structure as Ex1)."""
    l = logger.Analyzer(
        root="logs",
        folder_name=f"SOE",
        algorithm_name="SOE_uniform",
        algorithm_info="Exercise 3 single-objective (repair + diversity)"
    )
    
    for fid in PROBLEMS:
        print(f"\nRunning SOE on problem {fid}...")

        # Load and attach the problem
        problem = get_problem(fid=fid, instance=1, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l)

        # Run multiple population sizes and seeds
        for pop_size in POP_SIZES:
            for seed in range(1, RUNS + 1):
                print(f"  pop={pop_size}, run={seed}/{RUNS}", end="\r")
                soe_uniform(problem, seed, pop_size, BUDGET)
                problem.reset()
        print(f"  Completed all runs for problem {fid}")

    del l


def run_multi_objective():
    """Runs Exercise 3 Part 2 experiments and logs to IOHAnalyzer (same structure as Ex1)."""
            # Create logger once per problem
    l = logger.Analyzer(
        root="logs",
        folder_name=f"MOE",
        algorithm_name="MOE_uniform",
        algorithm_info="Exercise 3 multi-objective (NSGA-II style)"
    )
    
    for fid in PROBLEMS:
        print(f"\nRunning MOE on problem {fid}...")

        # Load and attach the problem
        problem = get_problem(fid=fid, instance=1, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l)

        # Run multiple population sizes and seeds
        for pop_size in POP_SIZES:
            for seed in range(1, RUNS + 1):
                print(f"  pop={pop_size}, run={seed}/{RUNS}", end="\r")
                moe_uniform(problem, seed, pop_size, BUDGET)
                problem.reset()
        print(f"  Completed all runs for problem {fid}")
    del l


# -------------------- MAIN ENTRY --------------------

if __name__ == "__main__":
    # You can run one or both depending on your needs:
    run_single_objective()
    run_multi_objective()
