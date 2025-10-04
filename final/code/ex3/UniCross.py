from ioh import get_problem, ProblemClass, logger
import numpy as np

# Genetic Algorithm (Uniform Crossover + Bit-Flip Mutation)
def genetic_algorithm(func, budget=None):
    n = func.meta_data.n_variables
    if budget is None:
        budget = int(50 * n * n)  # same evaluation budget as Exercise 2

    # --- GA parameters ---
    mu = 20            # parent population ≥10
    lambda_ = 40       # offspring per generation
    pc = 0.9           # crossover probability
    pm = 1.0 / n       # per-bit mutation probability
    k = 3              # tournament size

    # --- Initialise parent population ---
    pop = np.random.randint(2, size=(mu, n))
    fit = np.array([func(x) for x in pop])
    evals = mu

    # --- Selection, crossover, mutation ---
    def tournament_select():
        idx = np.random.randint(0, mu, k)
        return pop[idx[np.argmax(fit[idx])]].copy()

    def uniform_crossover(p1, p2):
        mask = np.random.rand(n) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def mutate(x):
        mask = np.random.rand(n) < pm
        x[mask] = 1 - x[mask]
        return x

    best_f = np.max(fit)
    best_x = pop[np.argmax(fit)].copy()

    # --- Main evolutionary loop ---
    while evals < budget:
        offspring = []
        while len(offspring) < lambda_:
            p1, p2 = tournament_select(), tournament_select()
            if np.random.rand() < pc:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(mutate(c1))
            if len(offspring) < lambda_:
                offspring.append(mutate(c2))

        offspring = np.array(offspring)
        off_fit = np.array([func(x) for x in offspring])
        evals += len(offspring)

        # --- (μ + λ) elitist replacement ---
        combined = np.vstack([pop, offspring])
        combined_fit = np.concatenate([fit, off_fit])
        idx = np.argsort(-combined_fit)[:mu]
        pop, fit = combined[idx], combined_fit[idx]

        if fit[0] > best_f:
            best_f = fit[0]
            best_x = pop[0].copy()

        # stop early if optimum reached
        if best_f >= func.optimum.y:
            break

    return best_f, best_x

# Problem Setup (same as Exercise 2)

om = get_problem(fid=1, dimension=50, instance=1, problem_class=ProblemClass.PBO)
lo = get_problem(fid=2, dimension=50, instance=1, problem_class=ProblemClass.PBO)
labs = get_problem(fid=18, dimension=50, instance=1, problem_class=ProblemClass.PBO)

# IOHprofiler Logger Setup

l = logger.Analyzer(
    root="data",
    folder_name="run",
    algorithm_name="genetic_algorithm",
    algorithm_info="Uniform crossover, bit-flip mutation, tournament selection, elitist replacement"
)

# Run GA on each problem (10 independent runs)

for problem in [om, lo, labs]:
    problem.attach_logger(l)
    for run in range(10):
        print(f"Running GA on problem {problem.meta_data.problem_id} run {run+1}/10 ...")
        f_opt, x_opt = genetic_algorithm(problem)
        print(f" → Run {run+1} best fitness = {f_opt}")
        problem.reset()

# Flush logger for IOHanalyzer
del l
print("GA runs complete — results saved.")
