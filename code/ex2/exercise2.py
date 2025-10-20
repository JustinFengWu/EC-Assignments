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
        # can delete later?
        if childReward > bestSoFar:
            bestSoFar = childReward
        if childCost <= k_limit and childReward > bestFeasible:
            bestFeasible = childReward
            
    print(f"Run complete: pid={problem}, seed={seed}")

def main():
    # run all required instances 30×
    
    problems = [2100,2101,2102,2103,2200,2201,2202,2203,2300,2301,2302]
    for problemId in problems:
        problem = get_problem(fid=problemId, instance=1, problem_class=ProblemClass.GRAPH)
        
        l = logger.Analyzer(
            root="ex2Data",  # top folder for all runs
            folder_name=f"f{problem}_run",
            algorithm_name="GSEMO",
            algorithm_info="Ex2 stuff")
        
        problem.attach_logger(l)
        for run in range(RUNS):
            gsemo_run(problem, seed=run+1)

if __name__ == "__main__":
    main()
