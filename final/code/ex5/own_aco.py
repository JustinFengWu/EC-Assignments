from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Own Ant Colony Optimization Algorithm
def own_aco_algorithm(func, budget, number_of_ants = 10, evaporation = 0.1, Q = 1.0):
    dimension = func.meta_data.n_variables
    pheromone = np.ones(dimension) * 0.5

    best_solution_so_far = None
    max_score_so_far = -np.inf
    evals = 0

    # Implementation of Local Search to improve the accuracy of results
    def local_search(solution, evals, budget):
        current = np.array(solution)
        current_score = func(current)
        evals += 1

        for i in range(dimension):
            if evals >= budget:
                break

            neighbor = current.copy()
            neighbor[i] = 1 - neighbor[i]
            score = func(neighbor)
            evals += 1

            if score > current_score:
                # Update the current_score with the maximum score so far
                current = neighbor
                current_score = score
        return current, current_score, evals

    # Termination Condition
    while evals < budget:
        iteration_solutions = []
        iteration_scores = []

        # Iterate on the given number of ants
        for _ in range(number_of_ants):
            if evals >= budget:
                break

            # Solution Construction from Pheromones
            solution = [1 if np.random.rand() < pheromone[i] else 0 for i in range(dimension)]

            solution, score, evals = local_search(solution, evals, budget) # Local Search Enhancements

            # Union with S (based on the pseudo code discussed in lectures)
            iteration_solutions.append(solution)
            iteration_scores.append(score)

            if score > max_score_so_far:
                max_score_so_far, best_solution_so_far = score, solution

        # Pheromone evaporation
        pheromone *= (1 - evaporation)

        if iteration_scores:
            best_idx = np.argmax(iteration_scores)
            for i in range(dimension):
                if iteration_solutions[best_idx][i]:
                    pheromone[i] += Q * iteration_scores[best_idx] / dimension # Deposit pheromone from best ant of this iteration

        # Keep pheromone values in the bound of [0.1, 0.9]
        pheromone = np.clip(pheromone, 0.1, 0.9)

    return max_score_so_far, best_solution_so_far

    

# Declaration of problems to be tested.
f1 = get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f2 = get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f3 = get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f18 = get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f23 = get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f24 = get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f25 = get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)


problems = [f1, f2, f3, f18, f23, f24, f25] # Running with all the function declarations

# Each algorithm gets its own logger so IOHanalyzer can separate them
for algo_name, algo_fn in [
    ("own_aco_algorithm", own_aco_algorithm),
]:
    l = logger.Analyzer(
        root="ex5/data",
        folder_name=algo_name,  # separate folder for each algorithm
        algorithm_name=algo_name,
        algorithm_info="exercise 5",
    )

    for f in problems:
        f.attach_logger(l)
        for r in range(10):
            algo_fn(f, 100000)  # budget = 100k evaluations
            f.reset()

    del l  # flush logger to disk
