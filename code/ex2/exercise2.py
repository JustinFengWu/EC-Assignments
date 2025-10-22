import random
import numpy as np
from ioh import get_problem, ProblemClass, logger

# here are where all the parameters are.
BUDGET = 10000
RUNS = 30

# For the First dot-point

def cost_of(x):
    return sum(x)

def dominates(a, b):
    rewardA, costA = a
    rewardB, costB = b
    # at least same then both, strictly better than at least one.
    if (rewardA >= rewardB and costA <= costB) and (rewardA > rewardB or costA < costB):
        return True
    return False

def mutate(individual, flipChance):
    x = len(individual)
    child = individual.copy()
    flippedOne = False
    while (not flippedOne) :
        for i in range(x):
            if random.random() < flipChance:
                child[i] ^= 1
                flippedOne = True
    return child

def insert_pareto(population, candidate):
    _, candidateReward, candidateCost = candidate
    # discard if dominated
    for _, reward, cost in population:
        if dominates((reward, cost), (candidateReward, candidateCost)):
            return False
        
    # remove dominated
    newPopulation = []
    for i in range(len(population)):
        individual, reward, cost = population[i]
        if (not dominates((candidateReward, candidateCost), (reward, cost))):
            newPopulation.append((individual, reward, cost))
    newPopulation.append(candidate)
    population[:] = newPopulation
    return True
    
    

def gsemo_run(problem, seed, k_limit=None):
    random.seed(seed)
    np.random.seed(seed)

    n = problem.meta_data.n_variables
    flipChance = 1.0 / n
    k_limit = round(0.1 * n)
    

    # the first solution is all zero
    individualInit = [0] * n
    rewardInit = problem(individualInit)
    costInit = cost_of(individualInit)
    population = []
    insert_pareto(population, (individualInit, rewardInit, costInit))

    evaluations = 1
    bestSoFar = rewardInit
    if (costInit <= k_limit):
        bestFeasible = rewardInit
    else:
        bestFeasible = float('-inf')

    # the loop where things evolve
    while evaluations < BUDGET:
        parentInd, _, _ = random.choice(population)
        child = mutate(parentInd, flipChance)
        childReward = problem(child)
        childCost = cost_of(child)
        evaluations += 1

        insert_pareto(population, (child, childReward, childCost))
        # can delete later
        if childReward > bestSoFar:
            bestSoFar = childReward
        if childCost <= k_limit and childReward > bestFeasible:
            bestFeasible = childReward
            
    print(f"Run complete: pid={problem.meta_data.problem_id}, seed={seed}")

def main():
    problems = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203, 2300, 2301, 2302]
    
    
            # Create a dedicated logger for this problem
    l = logger.Analyzer(
        root="ex2/data1",
        folder_name=f"GSEMO",
        algorithm_name="GSEMO",
        algorithm_info="Exercise 2 - GSEMO"
    )

    for problemId in problems:
        print(f"\nRunning GSEMO on problem {problemId}...")

        # Load and attach the problem
        problem = get_problem(fid=problemId, instance=1, problem_class=ProblemClass.GRAPH)
        problem.attach_logger(l)

        # Run 30 independent runs
        for run in range(RUNS):
            print(f"  Run {run+1}/{RUNS}", end="\r")
            gsemo_run(problem, seed=run+1)
            problem.reset()

        # Detach logger at the end of this problem
        print(f"  Completed all runs for problem {problemId}")
    del l
if __name__ == "__main__":
    main()