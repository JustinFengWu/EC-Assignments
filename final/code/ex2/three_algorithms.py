from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Please replace this `random search` by your `genetic algorithm`.
def random_search(func, budget):
    n = func.meta_data.n_variables
    f_opt = sys.float_info.min
    x_opt = None
    for i in range(budget):
        x = np.random.randint(2, size=n)
        f = func(x)
        if f > f_opt:
            f_opt, x_opt = f, x
    return f_opt, x_opt


def rls(func, budget):
    n = func.meta_data.n_variables

    initialSolution = np.random.randint(2, size=n)
    initialSolution_f = func(
        initialSolution
    )  # func() literally just evaluates the fitness of argument

    for _ in range(budget - 1):
        newSolution = initialSolution.copy()
        i = np.random.randint(n)
        newSolution[i] ^= 1  # flipping a random bit
        newSolution_f = func(newSolution)

        if newSolution_f >= initialSolution_f:
            initialSolution, initialSolution_f = newSolution, newSolution_f

    func.reset()
    return initialSolution_f, initialSolution


def one_plus_one_ea(func, budget):
    n = func.meta_data.n_variables
    p = (
        1.0 / n
    )  # the probabilty that determines whether each individual bit will flip or not

    initialSolution = np.random.randint(2, size=n)
    initialSolution_f = func(initialSolution)

    for _ in range(budget - 1):
        newSolution = initialSolution.copy()
        # generates an array of n elements, where each element is a number between 0-1
        flips = (
            np.random.rand(n) < p
        )  # compare each element with p, resulting in boolean array
        if flips.any():  # checks if any elements in flips is true
            newSolution[flips] ^= 1  # flip all relevant bits
        newSolution_f = func(newSolution)

        if newSolution_f >= initialSolution_f:
            initialSolution, initialSolution_f = newSolution, newSolution_f

    func.reset()
    return initialSolution_f, initialSolution


# Declaration of problems to be tested.
#  we can make om, lo, and labs array where fid = 1, 2 ..
f1 = get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f2 = get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f3 = get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f18 = get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f23 = get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f24 = get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO)
f25 = get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)


# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
# Put all problems in a list
problems = [f1, f2, f3, f18, f23, f24, f25]

# Each algorithm gets its own logger so IOHanalyzer can separate them
for algo_name, algo_fn in [
    ("random_search", random_search),
    ("rls", rls),
    ("one_plus_one_ea", one_plus_one_ea),
]:
    l = logger.Analyzer(
        root="ex2/data",
        folder_name=algo_name,  # separate folder for each algorithm
        algorithm_name=algo_name,
        algorithm_info="exercise 2",
    )

    for f in problems:
        f.attach_logger(l)
        for r in range(10):
            algo_fn(f, 100000)  # budget = 100k evaluations
            f.reset()

    del l  # flush logger to disk
