import numpy as np
from ioh import get_problem, ProblemClass, logger


def rls(func, budget):
    """Randomised Local Search - flips one random bit per iteration"""
    n = func.meta_data.n_variables
    
    current_solution = np.random.randint(2, size=n)
    current_fitness = func(current_solution)
    
    for _ in range(budget - 1):
        # flip a single random bit
        new_solution = current_solution.copy()
        i = np.random.randint(n)
        new_solution[i] ^= 1
        new_fitness = func(new_solution)
        
        # accept if better or equal (greedy)
        if new_fitness >= current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    
    return current_fitness, current_solution


def one_plus_one_ea(func, budget):
    """(1+1) EA - flips each bit independently with probability 1/n"""
    n = func.meta_data.n_variables
    p = 1.0 / n  # standard mutation rate
    
    current_solution = np.random.randint(2, size=n)
    current_fitness = func(current_solution)
    
    for _ in range(budget - 1):
        # mutate each bit with probability p
        new_solution = current_solution.copy()
        flips = np.random.rand(n) < p
        if flips.any():
            new_solution[flips] ^= 1
        new_fitness = func(new_solution)
        
        # accept if better or equal
        if new_fitness >= current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
    
    return current_fitness, current_solution


def genetic_algorithm(func, budget):
    """GA from Assignment 2 - uses tournament selection, crossover, and mutation"""
    n = func.meta_data.n_variables
    
    # parameters from assignment 2
    mu = 20            # population size
    lambda_ = 40       # offspring per generation
    pc = 0.9           # crossover probability
    pm = 1.0 / n       # mutation rate per bit
    k = 3              # tournament size
    
    # initialise random population
    pop = np.random.randint(2, size=(mu, n))
    fit = np.array([func(x) for x in pop])
    evals = mu
    
    def tournament_select():
        """Select parent using k-tournament selection"""
        idx = np.random.randint(0, mu, k)
        return pop[idx[np.argmax(fit[idx])]].copy()
    
    def uniform_crossover(p1, p2):
        """Uniform crossover - each bit randomly chosen from either parent"""
        mask = np.random.rand(n) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2
    
    def mutate(x):
        """Bit-flip mutation with probability pm per bit"""
        mask = np.random.rand(n) < pm
        x[mask] = 1 - x[mask]
        return x
    
    best_f = np.max(fit)
    best_x = pop[np.argmax(fit)].copy()
    
    # main evolutionary loop
    while evals < budget:
        offspring = []
        
        # generate lambda offspring
        while len(offspring) < lambda_:
            p1, p2 = tournament_select(), tournament_select()
            
            # crossover
            if np.random.rand() < pc:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # mutation
            offspring.append(mutate(c1))
            if len(offspring) < lambda_:
                offspring.append(mutate(c2))
        
        # evaluate offspring
        offspring = np.array(offspring)
        off_fit = np.array([func(x) for x in offspring])
        evals += len(offspring)
        
        # (mu+lambda) selection - keep best mu from parents and offspring
        combined = np.vstack([pop, offspring])
        combined_fit = np.concatenate([fit, off_fit])
        idx = np.argsort(-combined_fit)[:mu]
        pop, fit = combined[idx], combined_fit[idx]
        
        # track best solution found
        if fit[0] > best_f:
            best_f = fit[0]
            best_x = pop[0].copy()
    
    return best_f, best_x


# problem instances to test
problem_ids = [
    2100, 2101, 2102, 2103,  # MaxCoverage
    2200, 2201, 2202, 2203,  # MaxInfluence
    2300, 2301, 2302         # PackWhileTravel
]

algorithms = [
    ("RLS", rls),
    ("OnePlusOneEA", one_plus_one_ea),
    ("GeneticAlgorithm", genetic_algorithm)
]

budget = 10000  # function evaluations per run
num_runs = 30   # independent runs per instance

# run experiments
for algo_name, algo_func in algorithms:
    print(f"\nRunning {algo_name}...")
    
    # setup IOH logger
    l = logger.Analyzer(
        root="ex1/data",
        folder_name=algo_name,
        algorithm_name=algo_name,
        algorithm_info=f"Exercise 1"
    )
    
    for problem_id in problem_ids:
        print(f"Problem {problem_id}:")
        
        problem = get_problem(problem_id, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l)
        
        # run algorithm 30 times on this instance
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}", end='\r')
            
            best_fitness, best_solution = algo_func(problem, budget)
            problem.reset()
        
        print(f"  Completed {num_runs} runs")
    
    del l  # flush data to disk

print("\nAll done! Upload ex1/data to IOHanalyzer")