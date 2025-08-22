import random, math, statistics
from copy import deepcopy


# ---------------- TSP ----------------
class TSP:
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)

    @classmethod
    def from_tsplib(cls, path):
        coords, edge_type, in_coords = [], None, False
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line: 
                    continue
                if line.startswith("EDGE_WEIGHT_TYPE"):
                    edge_type = line.split(":")[-1].strip()
                if line == "NODE_COORD_SECTION":
                    in_coords = True; continue
                if line == "EOF":
                    break
                if in_coords:
                    parts = line.split()
                    if len(parts) >= 3:
                        _, x, y = parts[:3]
                        coords.append((float(x), float(y)))
        if edge_type not in (None, "EUC_2D"):
            raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}")
        return cls(coords)

    def distance(self, i, j):
        x1, y1 = self.coords[i]; x2, y2 = self.coords[j]
        return int(math.hypot(x1 - x2, y1 - y2) + 0.5)  # TSPLIB EUC_2D
    def tour_length(self, perm):
        return sum(self.distance(perm[i], perm[(i + 1) % self.n]) for i in range(self.n))
    
# ---------------- Individual ----------------
class Individual:
    def __init__(self, perm):
        self.perm = perm
        self.fitness = None

    @staticmethod
    def random(n):
        perm = list(range(n))
        random.shuffle(perm)
        return Individual(perm)

    def evaluate(self, tsp):
        self.fitness = tsp.tour_length(self.perm)

    def copy(self):
        c = Individual(self.perm[:])
        c.fitness = self.fitness
        return c

# ---------------- Operators ----------------
def inversion_mutation(perm):
    i, j = sorted(random.sample(range(len(perm)), 2))
    perm[i:j+1] = reversed(perm[i:j+1])   # <-- inclusive j


def swap_mutation(perm):
    i,j = random.sample(range(len(perm)),2)
    perm[i],perm[j] = perm[j],perm[i]

def order_crossover(p1, p2):
    n = len(p1)
    a,b = sorted(random.sample(range(n),2))
    child = [-1]*n
    child[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in child]
    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1
    return child

def edge_recombination(p1, p2):
    n = len(p1)
    neighbors = {c: set() for c in p1}
    for parent in (p1, p2):
        for i in range(n):
            c = parent[i]
            neighbors[c].add(parent[(i - 1) % n])
            neighbors[c].add(parent[(i + 1) % n])

    remaining = set(p1)
    current = random.choice(p1)
    child = []
    while remaining:
        child.append(current)
        remaining.remove(current)
        for v in neighbors.values():
            v.discard(current)
        cand = neighbors[current] & remaining
        if cand:
            m = min(len(neighbors[x]) for x in cand)
            current = random.choice([x for x in cand if len(neighbors[x]) == m])
        elif remaining:
            current = random.choice(tuple(remaining))
    return child


# ---------------- Selection ----------------
def tournament(pop, k=3):
    return min(random.sample(pop, k), key=lambda ind: ind.fitness)

# ---------------- EA Framework ----------------
def run_ea_generational(tsp, pop_size, gens, crossover, mutation, p_mut=None, ls_rate=0.0):
    if p_mut is None: p_mut = 1.0 / tsp.n
    pop = [Individual.random(tsp.n) for _ in range(pop_size)]
    for ind in pop: ind.evaluate(tsp)
    for _ in range(gens):
        elite = min(pop, key=lambda z: z.fitness).copy()
        newpop = [elite]
        while len(newpop) < pop_size:
            p1, p2 = tournament(pop), tournament(pop)
            child_perm = crossover(p1.perm, p2.perm)
            if random.random() < p_mut: mutation(child_perm)
            child = Individual(child_perm); child.evaluate(tsp)
            if ls_rate > 0 and random.random() < ls_rate:
                child = two_opt_local_search(tsp, child)
            newpop.append(child)
        pop = newpop
    return min(pop, key=lambda z: z.fitness)

def run_ea_generational_with_checkpoints(tsp, pop_size, gens, crossover, mutation, p_mut, ls_rate, cuts):
    pop = [Individual.random(tsp.n) for _ in range(pop_size)]
    for ind in pop: ind.evaluate(tsp)
    best_hist = {}
    next_cut_idx = 0
    cuts_sorted = sorted(cuts)
    for g in range(1, gens+1):
        # ... standard generational step ...
        elite = min(pop, key=lambda z: z.fitness).copy()
        newpop = [elite]
        while len(newpop) < pop_size:
            p1, p2 = tournament(pop), tournament(pop)
            child_perm = crossover(p1.perm, p2.perm)
            if random.random() < p_mut: mutation(child_perm)
            child = Individual(child_perm); child.evaluate(tsp)
            if ls_rate > 0 and random.random() < ls_rate:
                child = two_opt_local_search(tsp, child)
            newpop.append(child)
        pop = newpop

        # checkpoint
        while next_cut_idx < len(cuts_sorted) and g == cuts_sorted[next_cut_idx]:
            best_hist[g] = min(pop, key=lambda z: z.fitness).fitness
            next_cut_idx += 1
    return best_hist


def run_ea_steady_state(tsp, pop_size, iters, crossover, mutation, p_mut=None, ls_rate=0.0, replace=2):
    if p_mut is None: p_mut = 1.0 / tsp.n
    pop = [Individual.random(tsp.n) for _ in range(pop_size)]
    for ind in pop: ind.evaluate(tsp)
    for _ in range(iters):  # ~ gens*pop_size if you want parity
        kids = []
        for _ in range(replace):
            p1, p2 = tournament(pop), tournament(pop)
            child_perm = crossover(p1.perm, p2.perm)
            if random.random() < p_mut: mutation(child_perm)
            child = Individual(child_perm); child.evaluate(tsp)
            if ls_rate > 0 and random.random() < ls_rate:
                child = two_opt_local_search(tsp, child)
            kids.append(child)
        pop.sort(key=lambda z: z.fitness)            # best → worst
        pop[-replace:] = kids                         # replace worst
    return min(pop, key=lambda z: z.fitness)
def run_ea(tsp, pop_size, gens, crossover, mutation, p_mut=None, ls_rate=0.0):
    if gens <= 0: return run_ea_steady_state(tsp, pop_size, gens, crossover, mutation, p_mut, ls_rate)
    return run_ea_generational(tsp, pop_size, gens, crossover, mutation, p_mut, ls_rate)

def run_ea_steady_state_with_checkpoints(tsp, pop_size, gens, crossover, mutation,
                                         p_mut, ls_rate, cuts, replace=2):
    # parity: iters ≈ gens * pop_size / replace
    target_iters = (gens * pop_size) // replace
    cuts_sorted = sorted(cuts)
    cut_iters   = [(c, (c * pop_size) // replace) for c in cuts_sorted]

    pop = [Individual.random(tsp.n) for _ in range(pop_size)]
    for ind in pop: ind.evaluate(tsp)

    best_hist = {}
    next_idx = 0
    for it in range(1, target_iters + 1):
        # produce and insert 'replace' children
        kids = []
        for _ in range(replace):
            p1, p2 = tournament(pop), tournament(pop)
            child_perm = crossover(p1.perm, p2.perm)
            if random.random() < p_mut:
                mutation(child_perm)
            child = Individual(child_perm)
            child.evaluate(tsp)
            if ls_rate > 0 and random.random() < ls_rate:
                child = two_opt_local_search(tsp, child)
            kids.append(child)
        pop.sort(key=lambda z: z.fitness)      # best → worst
        pop[-replace:] = kids                  # replace worst

        # checkpoint when iteration count reaches parity for a generation cut
        while next_idx < len(cut_iters) and it == cut_iters[next_idx][1]:
            gen_cut = cut_iters[next_idx][0]
            best_hist[gen_cut] = min(pop, key=lambda z: z.fitness).fitness
            next_idx += 1
    return best_hist


# ---------------- 2-opt (for Memetic GA) ----------------
def two_opt_local_search(tsp, ind):
    perm = ind.perm[:]
    n = len(perm)
    best_len = ind.fitness if ind.fitness is not None else tsp.tour_length(perm)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                # delta evaluation (O(1))
                a, b = perm[(i - 1) % n], perm[i]
                c, d = perm[j], perm[(j + 1) % n]
                oldd = tsp.distance(a, b) + tsp.distance(c, d)
                newd = tsp.distance(a, c) + tsp.distance(b, d)
                if newd < oldd:
                    perm[i:j+1] = reversed(perm[i:j+1])
                    best_len += (newd - oldd)
                    improved = True
                    break
            if improved:
                break
    out = Individual(perm)
    out.fitness = best_len
    return out


# ---------------- Benchmark ----------------
GEN_CUTS = [2000, 5000, 10000, 20000]

def benchmark(instances, runs=30):
    import csv, statistics
    from collections import defaultdict
    vals = defaultdict(list)  # key: (instance, alg, pop, gen_cut) → list of bests

    for inst_path in instances:
        tsp = TSP.from_tsplib(inst_path)
        for pop in [20, 50, 100, 200]:
            for r in range(runs):
                seed = (hash(inst_path) ^ (pop << 8) ^ r) & 0xFFFFFFFF
                random.seed(seed)

                # EA-A: Generational (OX + inversion), checkpoints
                histA = run_ea_generational_with_checkpoints(
                    tsp, pop_size=pop, gens=max(GEN_CUTS),
                    crossover=order_crossover, mutation=inversion_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.0, cuts=GEN_CUTS
                )
                for cut in GEN_CUTS:
                    vals[(inst_path, "EA-A", pop, cut)].append(histA[cut])

                # EA-B: Steady-state (ERX + swap), checkpoints with parity
                histB = run_ea_steady_state_with_checkpoints(
                    tsp, pop_size=pop, gens=max(GEN_CUTS),
                    crossover=edge_recombination, mutation=swap_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.0, cuts=GEN_CUTS, replace=2
                )
                for cut in GEN_CUTS:
                    vals[(inst_path, "EA-B", pop, cut)].append(histB[cut])

                # EA-C: Memetic GA (generational + 2-opt on 20% offspring)
                histC = run_ea_generational_with_checkpoints(
                    tsp, pop_size=pop, gens=max(GEN_CUTS),
                    crossover=order_crossover, mutation=inversion_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.2, cuts=GEN_CUTS
                )
                for cut in GEN_CUTS:
                    vals[(inst_path, "EA-C", pop, cut)].append(histC[cut])

    # Write summary CSV
    with open("results/ea_benchmarks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "algorithm", "pop_size", "generations", "min", "mean", "std"])
        for k, arr in vals.items():
            inst, alg, pop, cut = k
            w.writerow([inst, alg, pop, cut,
                        min(arr), statistics.mean(arr), statistics.stdev(arr)])

def write_your_EA(instances, runs=30):
    import csv, statistics
    with open("results/your_EA.txt","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","avg_cost","stddev"])
        for inst in instances:
            tsp = TSP.from_tsplib(inst)
            scores = []
            for r in range(runs):
                random.seed((hash(inst) ^ r) & 0xFFFFFFFF)
                # Best EA: memetic (EA-C), pop=50, gens=20000
                hist = run_ea_generational_with_checkpoints(
                    tsp, pop_size=50, gens=20000,
                    crossover=order_crossover, mutation=inversion_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.2, cuts=[20000]
                )
                scores.append(hist[20000])
            w.writerow([inst, statistics.mean(scores), statistics.stdev(scores)])

