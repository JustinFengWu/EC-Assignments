import numpy as np
import random
import ioh
from ioh import logger
from ioh import ProblemClass
import math

# builds solution using pheromones
def construct_solution(pheromones, n):
    solution = np.zeros(n, dtype=int)
    
    for i in range(n):
        if random.random() < pheromones[i]:
            solution[i] = 1
    
    return solution

# updates pheromone values
def update_pheromones(pheromones, solution, rho, n):
    new_pheromones = pheromones.copy()
    
    lower_bound = 1.0 / n
    upper_bound = 1.0 - 1.0 / n
    
    for i in range(n):
        if solution[i] == 1:
            # reinforce
            new_pheromones[i] = min((1 - rho) * pheromones[i] + rho, upper_bound)
        else:
            # evaporate
            new_pheromones[i] =max((1 - rho) * pheromones[i], lower_bound)
    
    return new_pheromones

# MMAS algorithm (accepts equal fitness)
def mmas(problem, budget, rho):
    n = problem.meta_data.n_variables
    pheromones = np.full(n, 0.5)
    
    x_best = construct_solution(pheromones, n)
    f_best=problem(x_best)
    pheromones = update_pheromones(pheromones, x_best, rho, n)
    
    evals = 1
    
    while evals < budget:
        x = construct_solution(pheromones, n)
        f_x = problem(x)
        evals += 1
        
        if f_x >= f_best:  # accepts equal
            x_best = x.copy()
            f_best = f_x
        
        pheromones = update_pheromones(pheromones, x_best, rho, n)
    
    return x_best, f_best

# MMAS* algorithm (strict improvement only)
def mmas_star(problem, budget, rho):
    n = problem.meta_data.n_variables
    pheromones = np.full(n, 0.5)
    
    x_best = construct_solution(pheromones, n)
    f_best = problem(x_best)
    pheromones = update_pheromones(pheromones, x_best, rho, n)
    
    evals = 1
    while evals < budget:
        x = construct_solution(pheromones, n)
        f_x = problem(x)
        evals += 1
        
        if f_x > f_best:  # strict improvement
            x_best = x.copy()
            f_best = f_x
        
        pheromones = update_pheromones(pheromones, x_best, rho, n)
    
    return x_best, f_best

def run_exercise_4():
    n = 100
    budget = 100000
    runs = 10
    
    # load all problems
    f1 = ioh.get_problem(1, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f2 = ioh.get_problem(2, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f3 = ioh.get_problem(3, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f18 = ioh.get_problem(18, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f23 = ioh.get_problem(23, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f24 = ioh.get_problem(24, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    f25 = ioh.get_problem(25, dimension=n, instance=1, problem_class=ProblemClass.PBO)
    
    problems = [f1, f2, f3, f18, f23, f24, f25]
    
    # three rho values 
    rho_vals = [
        ("rho_1", 1.0),
        ("rho_sqrt_n", 1.0 / math.sqrt(n)),
        ("rho_1_over_n", 1.0 / n)
    ]
    
    print("Starting experiments for exercise 4")
    print("n =", n, "budget =", budget, "runs =", runs)
    
    for rho_name, rho_val in rho_vals:
        print("\nTesting rho =", rho_val, "(" + rho_name + ")")
        
        # run MMAS
        log1 = logger.Analyzer(
            root="ex4/data",
            folder_name="mmas_" + rho_name,
            algorithm_name="mmas_" + rho_name,
            algorithm_info="exercise 4"
        )
        
        for prob in problems:
            prob.attach_logger(log1)
            print("  MMAS on", prob.meta_data.name, "... ", end="")
            for run in range(runs):
                prob.reset()
                mmas(prob, budget, rho_val)
            print("done")
        
        del log1
  
        # run MMAS*
        log2 = logger.Analyzer(
            root="ex4/data", 
            folder_name="mmas_star_" + rho_name,
            algorithm_name="mmas_star_" + rho_name,
            algorithm_info="exercise 4"
        )
        
        for prob in problems:
            prob.attach_logger(log2)
            print("  MMAS* on", prob.meta_data.name, "... ", end="")
            for run in range(runs):
                prob.reset()
                mmas_star(prob, budget, rho_val)
            print("done")
        
        del log2
    
    print("\nCompleted")

# quick test
def test_algorithms():
    print("Testing algorithms...")
    
    test_prob = ioh.get_problem(1, dimension=20, instance=1, problem_class=ProblemClass.PBO)
    
    test_prob.reset()
    _, result1 = mmas(test_prob, 1000, 0.1)
    print("MMAS:", result1, "/20")
    
    test_prob.reset()
    _, result2 = mmas_star(test_prob, 1000, 0.1)
    print("MMAS*:", result2, "/20")
    
    if result1 >= 15 and result2 >= 15:
        print("Test OK")
        return True
    else:
        print("Results lower than expected")
        return False

if __name__ == "__main__":
    if test_algorithms():
        print()
        answer = input("Run full experiments? (y/n): ")
        if answer.lower() == 'y':
            run_exercise_4()
