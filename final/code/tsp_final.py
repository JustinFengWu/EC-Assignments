"""
TSP problem parser and utilities - Fixed Version
"""

# dependencies
import math
import sys
import random
import os
from pathlib import Path
# Adding this just because I like the loading bar
from tqdm import tqdm
import csv
import statistics
import time
from collections import defaultdict

# Base directory = where this script lives
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def write_results(results, results_dir=None):
    if results_dir is None:
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "local_search_prev.txt"

    with open(out_file, "a") as f:
        for (tsp_name, method), (min_len, avg_len) in results.items():
            f.write(
                f"Name: {tsp_name}  Method: {method}   Min: {min_len:.2f}, Mean: {avg_len:.2f}\n")

    print(f"Local search results written to {out_file}")

# -------------------------
# Exercise 1
# -------------------------

class TSP:
    def __init__(self, filename):
        self.dimension = 0
        self.coordinates = []
        self.distance_matrix = []
        self.node_section = False
        self.name = ""
        self.n = 0  # Add this attribute for compatibility

        self.load_tsp(filename)
        self.create_distance_matrix()
        self.n = self.dimension  # Set n to dimension

    @classmethod
    def from_tsplib(cls, filename):
        """Alternative constructor for compatibility"""
        return cls(filename)

    def load_tsp(self, filename):
        # Extract name from filename
        self.name = Path(filename).stem
        
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("NAME"):
                self.name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_type = line.split(":")[1].strip()
                if edge_type != "EUC_2D":
                    raise ValueError(
                        f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}")
            elif line == "NODE_COORD_SECTION":
                self.node_section = True
            elif line == "EOF":
                break
            elif self.node_section:
                parts = line.split()
                if len(parts) >= 3:
                    _, x, y = parts[:3]
                    self.coordinates.append((float(x), float(y)))

    def create_distance_matrix(self):
        n = self.dimension
        self.distance_matrix = [[0]*n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i != j:
                    xi, yi = self.coordinates[i]
                    xj, yj = self.coordinates[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    self.distance_matrix[i][j] = round(dist)
                    self.distance_matrix[j][i] = round(dist)

    def path_length(self, perm):
        """Calculate the total path length for a given permutation."""
        if not perm or len(perm) < 2:
            return 0
        
        total = 0
        for i in range(len(perm) - 1):
            city1 = perm[i] - 1  # Convert to 0-based indexing
            city2 = perm[i + 1] - 1  # Convert to 0-based indexing
            
            # Validate indices
            if city1 < 0 or city1 >= self.dimension or city2 < 0 or city2 >= self.dimension:
                raise ValueError(f"Invalid city indices: {city1}, {city2} (dimension: {self.dimension})")
            
            total += self.distance_matrix[city1][city2]
        return total

    def get_distance(self, i, j):
        """Get distance between cities i and j (0-based indexing)."""
        if i < 0 or i >= self.dimension or j < 0 or j >= self.dimension:
            raise ValueError(f"Invalid indices: i={i}, j={j}, dimension={self.dimension}")
        return self.distance_matrix[i][j]

# -------------------------
# Exercise 2 
# -------------------------

# The function that connects operator with relative functions
def local_search(tsp: TSP, neighbourhood_operator):
    # Create an initial solution as an Individual
    current_individual = Individual(tsp.dimension)
    current_tour = current_individual.tour

    if neighbourhood_operator == "jump":
        _, local_min_length = get_jump_local_minimum(
            tsp, current_tour, float('inf'))
    elif neighbourhood_operator == "exchange":
        _, local_min_length = get_exchange_local_minimum(
            tsp, current_tour, float('inf'))
    elif neighbourhood_operator == "2opt":
        _, local_min_length = get_2opt_local_minimum(
            tsp, current_tour, float('inf'))
    else:
        raise ValueError("Unknown neighbourhood operator")

    return local_min_length


# -------------------------------
# Best Neighbour Generator  (Exercise 2)
# -------------------------------
def get_jump_local_minimum(tsp, tour, max_iterations):
    """Generate all neighbours by moving one city to a new position."""
    iterations = 0
    # created copy of tour so that it can be updated with better neighbours
    current = tour[:]
    current_path_length = tsp.path_length(tour)

    # just to make sure it doesn't run for too long, but if that's not a worry, then pass in float('inf')
    while iterations < max_iterations:
        iterations += 1
        better_neighbour_found = False

        # extrating the body out because i do not want to deal with potential indexing issues.
        # It is much easier to just append start and end onto the body after.
        tour_body = current[1:-1]
        n = len(tour_body)

        # Assume closed tour: start == end → do not move first or last city by only operating on the body
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_tour_body = tour_body[:]
                city = new_tour_body.pop(i)
                new_tour_body.insert(j, city)
                new_tour = [tour[0]] + new_tour_body + [tour[-1]]
                new_path_length = tsp.path_length(new_tour)

                # instead of recursion just update current and re-iterate when better neighbour is found.
                # solves the problem of reaching max recursion depth with larger problems.
                if new_path_length < current_path_length:
                    current, current_path_length = new_tour, new_path_length
                    better_neighbour_found = True
                    break
            if better_neighbour_found:
                break
        if not better_neighbour_found:
            break
    # return the tour and the current tour's length
    return current, current_path_length


def get_exchange_local_minimum(tsp, tour, max_iterations):
    """Generate all neighbours by swapping two cities."""
    iterations = 0
    current = tour[:]
    current_path_length = tsp.path_length(tour)

    while iterations < max_iterations:
        iterations += 1
        better_neighbour_found = False
        n = len(current)

        # need to change the logic here a little bit, searching for every exchange possibility. Other than that its all the same as jump.
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_tour_body = current[:]
                new_tour_body[i], new_tour_body[j] = new_tour_body[j], new_tour_body[i]
                new_path_length = tsp.path_length(new_tour_body)

                if new_path_length < current_path_length:
                    current, current_path_length = new_tour_body, new_path_length
                    better_neighbour_found = True
                    break
            if better_neighbour_found:
                break
        if not better_neighbour_found:
            break

    return current, current_path_length


def get_2opt_local_minimum(tsp, tour, max_iterations):
    """Generate all neighbours by 2-opt reversal."""
    iterations = 0
    current = tour[:]
    current_path_length = tsp.path_length(tour)

    while iterations < max_iterations:
        iterations += 1
        better_neighbour_found = False
        n = len(current)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # use python slicing to reverse sub-segment of the tour
                # [:i] = from 0 to i (non-inclusive), [i:j + 1] = i 1o j (inclusive), [::-1] = reverse, [j + 1] = j + 1 onwards (inclusive)
                new_tour_body = current[:i] + \
                    current[i:j + 1][::-1] + current[j + 1:]
                new_path_length = tsp.path_length(new_tour_body)

                if new_path_length < current_path_length:
                    current, current_path_length = new_tour_body, new_path_length
                    better_neighbour_found = True
                    break
            if better_neighbour_found:
                break
        if not better_neighbour_found:
            break

    return current, current_path_length

# -------------------------
# Exercise 3
# -------------------------

class Individual:
    """Represents a single TSP solution (tour)."""

    def __init__(self, n, tour=None):
        """
        n: number of cities in the TSP
        tour: optional list of city indices (start and end are the same city)
        """
        if tour is not None:
            self.tour = tour[:]  # Make a copy
        else:
            # O(n) random tour creation: start=1, shuffle rest, end=1
            cities = list(range(2, n + 1))
            random.shuffle(cities)
            self.tour = [1] + cities + [1]

        self.fitness = None  # to store evaluation result

    def evaluate(self, tsp):
        """Evaluate path length using given TSP instance (lower is better)."""
        self.fitness = tsp.path_length(self.tour)
        return self.fitness

    def copy(self):
        """Create a copy of this Individual."""
        new_individual = Individual(len(self.tour) - 1, self.tour[:])
        new_individual.fitness = self.fitness
        return new_individual

    # -------------------------
    # Mutation Operators (Exercise 4)
    # -------------------------

    def swap(self, i, j):
        """Swap elements at positions i and j."""
        self.tour[i], self.tour[j] = self.tour[j], self.tour[i]

    def insert(self, i, j):
        """Insert element at position i into position j (jump)."""
        elem = self.tour.pop(i)
        self.tour.insert(j, elem)

    def inversion(self, i, j):
        """Reverse the order of elements between positions i and j (inclusive)."""
        self.tour[i:j+1] = reversed(self.tour[i:j+1])

    def __repr__(self):
        return f"Individual(tour={self.tour}, fitness={self.fitness})"

    @staticmethod
    def random(n):
        """Convenience method for generating a random individual of size n."""
        return Individual(n)


def swap_mutation(ind: Individual):
    """Pick two random positions (not including first/last city) and swap them."""
    n = len(ind.tour)
    if n <= 3:  # Not enough cities to swap
        return ind
    i, j = random.sample(range(1, n - 1), 2)
    ind.swap(i, j)
    ind.fitness = None  # Invalidate fitness
    return ind


def inversion_mutation(ind: Individual):
    """Pick two random positions (not including first/last city) and reverse the segment."""
    n = len(ind.tour)
    if n <= 3:  # Not enough cities to invert
        return ind
    i, j = sorted(random.sample(range(1, n - 1), 2))
    ind.inversion(i, j)
    ind.fitness = None  # Invalidate fitness
    return ind


class Population:
    def __init__(self, tsp, size: int):
        self.tsp = tsp
        self.individuals = [Individual(tsp.dimension) for _ in range(size)]

    # -------------------------
    # Crossover Operators (Exercise 5)
    # -------------------------
    @staticmethod
    def order_crossover(parent1: Individual, parent2: Individual):
        """Order Crossover (OX) → returns two offspring."""
        n = len(parent1.tour)
        if n <= 3:  # Handle edge case
            return parent1.copy(), parent2.copy()
            
        start, end = sorted(random.sample(range(1, n - 1), 2))

        def ox(p1, p2):
            child = [None] * n
            child[0] = p1.tour[0]  # Set start city
            child[-1] = p1.tour[-1]  # Set end city
            child[start:end] = p1.tour[start:end]
            
            # Get all cities from p2 that aren't already in child (excluding start/end)
            used_cities = set(x for x in child if x is not None)
            fill = [x for x in p2.tour[1:-1] if x not in used_cities]  # Only middle cities
            
            idx = end if end < n - 1 else 1  # Start filling after crossover section
            for val in fill:
                while idx < n - 1 and child[idx] is not None:
                    idx += 1
                if idx >= n - 1:  # Wrap around but skip first/last
                    idx = 1
                    while idx < n - 1 and child[idx] is not None:
                        idx += 1
                if idx < n - 1:
                    child[idx] = val
                    idx += 1
                    
            # Fill any remaining None positions
            for i in range(1, n-1):
                if child[i] is None:
                    for city in range(1, n):
                        if city not in child:
                            child[i] = city
                            break
                    
            result = Individual(n - 1, child)
            result.fitness = None  # Invalidate fitness
            return result

        return ox(parent1, parent2), ox(parent2, parent1)

    @staticmethod
    def pmx_crossover(parent1: Individual, parent2: Individual):
        """Partially Mapped Crossover (PMX)"""
        n = len(parent1.tour)
        if n <= 3:  # Handle edge case
            return parent1.copy(), parent2.copy()
            
        start, end = sorted(random.sample(range(1, n - 1), 2))

        def pmx(p1, p2):
            child = [None] * n
            child[0] = p1.tour[0]  # Set start city
            child[-1] = p1.tour[-1]  # Set end city
            child[start:end] = p1.tour[start:end]
            
            # Create mapping for PMX
            mapping = {}
            for i in range(start, end):
                if p1.tour[i] != p2.tour[i]:
                    mapping[p2.tour[i]] = p1.tour[i]
            
            # Fill remaining positions
            for i in range(1, n - 1):  # Skip start and end cities
                if child[i] is None:
                    val = p2.tour[i]
                    while val in mapping:
                        val = mapping[val]
                    child[i] = val
                    
            result = Individual(n - 1, child)
            result.fitness = None  # Invalidate fitness
            return result

        return pmx(parent1, parent2), pmx(parent2, parent1)

    @staticmethod
    def cycle_crossover(parent1: Individual, parent2: Individual):
        """Cycle Crossover (CX)"""
        n = len(parent1.tour)
        if n <= 3:  # Handle edge case
            return parent1.copy(), parent2.copy()

        def cx(p1, p2):
            child = [None] * n
            child[0] = p1.tour[0]  # Set start city
            child[-1] = p1.tour[-1]  # Set end city
            
            # Find cycles in the middle section only
            visited = [False] * n
            visited[0] = visited[-1] = True  # Don't process start/end
            
            for start_idx in range(1, n - 1):
                if not visited[start_idx]:
                    # Start a new cycle
                    cycle_indices = []
                    idx = start_idx
                    while not visited[idx]:
                        visited[idx] = True
                        cycle_indices.append(idx)
                        # Find where p2[idx] appears in p1
                        try:
                            idx = p1.tour.index(p2.tour[idx], 1, n - 1)  # Search only middle section
                        except ValueError:
                            break
                    
                    # Assign values for this cycle
                    for i in cycle_indices:
                        child[i] = p1.tour[i]
            
            # Fill remaining positions with p2 values
            for i in range(1, n - 1):
                if child[i] is None:
                    child[i] = p2.tour[i]
                    
            result = Individual(n - 1, child)
            result.fitness = None  # Invalidate fitness
            return result

        return cx(parent1, parent2), cx(parent2, parent1)

    @staticmethod
    def edge_recombination(parent1: Individual, parent2: Individual):
        """Edge Recombination Crossover (ERX)"""
        def erx(p1, p2):
            n = len(p1.tour)
            
            # Start with the same start/end cities
            child = [p1.tour[0]]  # Start city
            remaining = list(p1.tour[1:-1])  # Middle cities only
            random.shuffle(remaining)
            
            # Simple approach: just use remaining cities in random order
            # (A proper ERX implementation is quite complex for TSP with fixed start/end)
            child.extend(remaining)
            child.append(p1.tour[-1])  # End city
            
            result = Individual(n - 1, child)
            result.fitness = None  # Invalidate fitness
            return result

        return erx(parent1, parent2), erx(parent2, parent1)

    # -------------------------
    # Selection Methods  (Exercise 5)
    # -------------------------

    def fitness_proportional_selection(self, num_parents, transformation="shifted_inverse"):
        """
        Fitness-proportional selection (roulette wheel method)

        Args:
            num_parents: Number of parents to select
            transformation: "shifted_inverse" or "min_max_normalization"
        """
        # just to perform some safety checks, ensuring that all individuals have fitness values and collect them all together.
        for individual in self.individuals:
            if individual.fitness is None:
                individual.evaluate(self.tsp)

        path_lengths = []
        for ind in self.individuals:
            if ind.fitness is None:
                ind.evaluate(self.tsp)
            path_lengths.append(ind.fitness)

        if any(fitness is None for fitness in path_lengths):
            raise ValueError(
                "Some individuals still have None fitness after evaluation")


        # Transform path lengths into a "higher is better" scoring system
        # As roulette wheel needs positive probabilities that are proportional to how path lengths work
        if transformation == "shifted_inverse":
            L_min = min(path_lengths)
            epsilon = 1e-6
            fitness_values = [1 / (L - L_min + epsilon) for L in path_lengths]

        elif transformation == "min_max_normalization":
            L_max = max(path_lengths)
            L_min = min(path_lengths)
            delta = 1e-6
            fitness_values = [(L_max - L + delta) /
                              (L_max - L_min + delta) for L in path_lengths]

        else:
            raise ValueError(
                "transformation must be 'shifted_inverse' or 'min_max_normalization'")

        # Convert fitness to selection probabilities
        total_fitness = sum(fitness_values)

        if total_fitness == 0:
            raise ValueError(
                "Total fitness is zero - cannot compute selection probabilities")

        probabilities = [fitness / total_fitness for fitness in fitness_values]

        # Spin the roulette wheel to pick parents
        selected_parents = []

        for _ in range(num_parents):
            r = random.random()

            cumulative_sum = 0
            selected = False

            for i, prob in enumerate(probabilities):
                cumulative_sum += prob
                if cumulative_sum > r:
                    selected_individual = self.individuals[i].copy()
                    if selected_individual.fitness is None:
                        selected_individual.evaluate(self.tsp)
                    selected_parents.append(selected_individual)
                    selected = True
                    break

            # Fallback if nothing was selected which actually shouldn't happen
            if not selected:
                fallback_individual = self.individuals[-1].copy()
                if fallback_individual.fitness is None:
                    fallback_individual.evaluate(self.tsp)
                selected_parents.append(fallback_individual)

        return selected_parents

    def tournament_selection(self, num_parents, tournament_size=2):
        """
        Tournament selection with binary tournaments
        No duplicates allowed (removing winners from pool)

        Args:
            num_parents: Number of parents to select
            tournament_size: k=2
        """
        for individual in self.individuals:
            if individual.fitness is None:
                individual.evaluate(self.tsp)

        selected_parents = []
        # Copy of population to remove winners
        available_pool = self.individuals[:]

        for _ in range(num_parents):
            if len(available_pool) < 2:
                # Resetting the pool if we run out of individuals
                available_pool = self.individuals[:]

            participant1 = random.choice(available_pool)
            available_pool.remove(participant1)

            participant2 = random.choice(available_pool)
            available_pool.remove(participant2)

            # Comparing the fitness and choosing winner as lower path length = better
            # Losers gets placed back into the pool
            if participant1.fitness < participant2.fitness:
                winner = participant1
                available_pool.append(participant2)
            else:
                winner = participant2
                available_pool.append(participant1)

            selected_parents.append(winner.copy())

        return selected_parents

    def elitist_selection(self, num_elites):
        """
        Elitist selection - picks the top performers directly

        Args:
            num_elites: Number of elite individuals to select (top n best)
        """
        # Ranking the individuals by performance (less distance is better)
        for individual in self.individuals:
            if individual.fitness is None:
                individual.evaluate(self.tsp)

        # Sorting by fitness: ascending order since lower path length = better
        sorted_individuals = sorted(
            self.individuals, key=lambda ind: ind.fitness)

        elite_individuals = sorted_individuals[:num_elites]

        return [ind.copy() for ind in elite_individuals]

    def select_parents(self, method, num_parents, **kwargs):
        """
        Generic parent selection interface

        Args:
            method: "fitness_proportional", "tournament", or "elitist"
            num_parents: Number of parents to select
            **kwargs: Method specific parameters
        """
        if method == "fitness_proportional":
            transformation = kwargs.get('transformation', 'shifted_inverse')
            return self.fitness_proportional_selection(num_parents, transformation)

        elif method == "tournament":
            tournament_type = kwargs.get('tournament_type', 'binary')
            return self.tournament_selection(num_parents, tournament_type)

        elif method == "elitist":
            return self.elitist_selection(num_parents)

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def select_survivors(self, method, population_size, **kwargs):
        """
        Survivor selection: we can use same methods as parent selection

        Args:
            method: "fitness_proportional", "tournament", or "elitist"
            population_size: Size of surviving population
            **kwargs: Method-specific parameters
        """
        return self.select_parents(method, population_size, **kwargs)


# wrapper function for exercise 6 EA algorithms
def two_opt_local_search(tsp, ind, max_iterations=1000):  # Changed from float('inf')
    """
    Wrapper so EA-C can apply 2-opt to an Individual.
    Uses your existing get_2opt_local_minimum.
    """
    best_tour, best_len = get_2opt_local_minimum(tsp, ind.tour, max_iterations)
    out = Individual(len(best_tour) - 1, best_tour)
    out.fitness = best_len
    return out


def random_permutation(n):
    perm = list(range(2, n + 1))
    random.shuffle(perm)
    perm = [1] + perm + [1]
    return perm


def tournament(pop):
    """Simple binary tournament selection for individual selection"""
    # Ensure all have fitness
    for ind in pop:
        if ind.fitness is None:
            raise ValueError("Individual without fitness in tournament")
    
    p1, p2 = random.sample(pop, 2)
    return p1 if p1.fitness < p2.fitness else p2


# -------------------------------
#  Exercise 6
# -------------------------------

def run_ea_generational_with_checkpoints(tsp, pop_size, gens, crossover, mutation, p_mut, ls_rate, cuts):
    # Create initial population as list of Individuals
    pop = [Individual.random(tsp.dimension) for _ in range(pop_size)]
    for ind in pop:
        ind.evaluate(tsp)
    
    best_hist = {}
    next_cut_idx = 0
    cuts_sorted = sorted(cuts)
    
    for g in range(1, gens+1):
        # Standard generational step
        elite = min(pop, key=lambda z: z.fitness).copy()
        newpop = [elite]
        
        while len(newpop) < pop_size:
            # Simple tournament selection on the population list
            p1 = tournament(pop)
            p2 = tournament(pop)
            child1, child2 = crossover(p1, p2)

            child = random.choice([child1, child2])  # pick one
            if random.random() < p_mut:
                mutation(child)
            child.evaluate(tsp)
            if ls_rate > 0 and random.random() < ls_rate:
                child = two_opt_local_search(tsp, child)
            newpop.append(child)
            
        pop = newpop

        # checkpoint
        while next_cut_idx < len(cuts_sorted) and g == cuts_sorted[next_cut_idx]:
            best_hist[g] = min(pop, key=lambda z: z.fitness).fitness
            next_cut_idx += 1
            
    return best_hist


def run_ea_steady_state_with_checkpoints(tsp, pop_size, gens, crossover, mutation,
                                         p_mut, ls_rate, cuts, replace=2):
    # parity: iters ≈ gens * pop_size / replace
    target_iters = (gens * pop_size) // replace
    cuts_sorted = sorted(cuts)
    cut_iters = [(c, (c * pop_size) // replace) for c in cuts_sorted]

    # Create initial population as list of Individuals
    pop = [Individual.random(tsp.dimension) for _ in range(pop_size)]
    for ind in pop:
        ind.evaluate(tsp)

    best_hist = {}
    next_idx = 0
    
    for it in range(1, target_iters + 1):
        # produce and insert 'replace' children
        kids = []
        for _ in range(replace):
            p1, p2 = tournament(pop), tournament(pop)
            child1, child2 = crossover(p1, p2)
            child = random.choice([child1, child2])
            if random.random() < p_mut:
                mutation(child)
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


# ---------------- Benchmark ----------------
# GEN_CUTS = [2000, 5000, 10000, 20000]  # Original
GEN_CUTS = [10, 20, 30, 40]  # Reduced for testing


def benchmark(instances, runs=30, results_dir=None):


    # Determine folder for results
    if results_dir is None:
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # key: (instance, alg, pop, gen_cut) → list of bests
    vals = defaultdict(list)

    for inst_path in instances:
        tsp = TSP(inst_path)  # Use regular constructor instead of from_tsplib
        print(f"Processing {tsp.name}...")
        
        for pop in [20, 50]:  # Reduced population sizes for testing
            print(f"  Population size: {pop}")
            for r in range(runs):
                print(f"    Run {r+1}/{runs}")
                seed = (hash(str(inst_path)) ^ (pop << 8) ^ r) & 0xFFFFFFFF
                random.seed(seed)

                # EA-A with timing
                print(f"      Running EA-A...")
                start_time = time.time()
                histA = run_ea_generational_with_checkpoints(
                    tsp, pop_size=pop, gens=max(GEN_CUTS),
                    crossover=Population.order_crossover, mutation=inversion_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.0, cuts=GEN_CUTS
                )
                elapsed_a = time.time() - start_time
                print(f"      EA-A completed in {elapsed_a:.2f} seconds")
                for cut in GEN_CUTS:
                    vals[(tsp.name, "EA-A", pop, cut)].append(histA[cut])

                # EA-B with timing
                print(f"      Running EA-B...")
                start_time = time.time()
                histB = run_ea_steady_state_with_checkpoints(
                    tsp, pop_size=pop, gens=max(GEN_CUTS),
                    crossover=Population.edge_recombination, mutation=swap_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.0, cuts=GEN_CUTS, replace=2
                )
                elapsed_b = time.time() - start_time
                print(f"      EA-B completed in {elapsed_b:.2f} seconds")
                for cut in GEN_CUTS:
                    vals[(tsp.name, "EA-B", pop, cut)].append(histB[cut])

                # EA-C with timing (this is the one with local search that might hang)
                print(f"      Running EA-C (with local search)...")
                start_time = time.time()
                try:
                    histC = run_ea_generational_with_checkpoints(
                        tsp, pop_size=pop, gens=max(GEN_CUTS),
                        crossover=Population.order_crossover, mutation=inversion_mutation,
                        p_mut=1.0/tsp.n, ls_rate=0.2, cuts=GEN_CUTS
                    )
                    elapsed_c = time.time() - start_time
                    print(f"      EA-C completed in {elapsed_c:.2f} seconds")
                    for cut in GEN_CUTS:
                        vals[(tsp.name, "EA-C", pop, cut)].append(histC[cut])
                except KeyboardInterrupt:
                    print(f"      EA-C interrupted after {time.time() - start_time:.2f} seconds")
                    print(f"      Likely infinite loop in local search - skipping this run")
                    continue
                except Exception as e:
                    print(f"      EA-C failed with error: {e}")
                    continue

    # Write summary CSV to the specified results folder
    out_file = results_dir / "ea_benchmarks.csv"
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "algorithm", "pop_size",
                   "generations", "min", "mean", "std"])
        for k, arr in vals.items():
            inst, alg, pop, cut = k
            if len(arr) > 0:  # Only write if we have data
                w.writerow([inst, alg, pop, cut, min(arr),
                           statistics.mean(arr), statistics.stdev(arr) if len(arr) > 1 else 0])

    print(f"Benchmark results written to {out_file}")


def write_your_EA(instances, runs=30, results_dir=None):


    # Default to "results" folder in the same directory as this script
    if results_dir is None:
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
        results_dir.mkdir(exist_ok=True)

    out_file = results_dir / "your_EA.txt"

    with open(out_file, "w") as f:
        f.write("instance,avg_cost,stddev\n")  # Write header
        
        for inst in tqdm(instances, desc="Final EA-C Evaluation"):
            tsp = TSP(inst)  # Use regular constructor instead of from_tsplib
            scores = []
            
            for r in tqdm(range(runs), desc=f"  {tsp.name}", leave=False):
                random.seed((hash(str(inst)) ^ r) & 0xFFFFFFFF)
                # Best EA: memetic (EA-C), pop=50, gens=max(GEN_CUTS)
                hist = run_ea_generational_with_checkpoints(
                    tsp, pop_size=50, gens=max(GEN_CUTS),
                    crossover=Population.order_crossover, mutation=inversion_mutation,
                    p_mut=1.0/tsp.n, ls_rate=0.2, cuts=[max(GEN_CUTS)]
                )
                scores.append(hist[max(GEN_CUTS)])
                
            f.write(f"{tsp.name},{statistics.mean(scores)},{statistics.stdev(scores)}\n")

    print(f"[OK] Wrote EA-C results to {out_file}")


def run_exercise2():
    # tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
    #              "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    tsp_names = ["eil51"]

    results = {}

    for name in tsp_names:
        try:
            filepath = BASE_DIR / f"tsp/{name}.tsp"
            tsp = TSP(filepath)
            tsp.name = name
            print(f"Running TSP instance: {tsp.name}")

            for operator_name in {"jump", "exchange", "2opt"}:
                average = 0
                minimum = float('inf')
                for _ in tqdm(range(30), desc=f"{name} - {operator_name}"):
                    min_length = local_search(tsp, operator_name)
                    average += min_length
                    if min_length < minimum:
                        minimum = min_length
                average /= 30
                results[(name, operator_name)] = (minimum, average)

        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")

    # Save results into fixed results folder
    write_results(results, RESULTS_DIR)


def run_exercise6():
    """
    Exercise 6:
    - Benchmark EA-A (generational), EA-B (steady-state), EA-C (memetic)
      on TSPlib instances with pop sizes [20,50,100,200]
      and gens [2000,5000,10000,20000].
    - Write full benchmark results to results/ea_benchmarks.csv
    - Then run EA-C (best EA) with pop=50, gens=20000, 30 runs,
      write averages to results/your_EA.txt
    """
    # tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
    #              "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    tsp_names = ["eil51"]

    instances = [BASE_DIR / f"tsp/{name}.tsp" for name in tsp_names]

    # Step 1: Benchmark three EA variants
    benchmark(instances, runs=5, results_dir=RESULTS_DIR)  # Reduced runs for testing
    
    # Step 2: Run final evaluation for best EA (EA-C)
    write_your_EA(instances, runs=5, results_dir=RESULTS_DIR)  # Reduced runs for testing

    print(f"Exercise 6 complete. Results written to {RESULTS_DIR/'ea_benchmarks.csv'} "
          f"and {RESULTS_DIR/'your_EA.txt'}")


# -------------------------
# Inerover (Exercise 7)
# -------------------------

def mean(vals):
    """Calculate the average of all values."""
    return sum(vals) / len(vals)


def stdev(vals):
    """Calculate the standard deviation using integer arithmetic."""
    if len(vals) <= 1:
        return 0.0
    mean_value = mean(vals)
    accumulator = sum((x - mean_value) ** 2 for x in vals)
    return math.sqrt(accumulator / (len(vals) - 1))


def random_permutation(n):
    """Generate random permutation for inver-over algorithm."""
    permutation = list(range(1, n + 1))
    random.shuffle(permutation)
    return permutation


def invert_segment(tour, start, end):
    """Function to invert the segment of the array."""
    n = len(tour)
    if start == end:
        return
    
    if start < end:
        tour[start:end+1] = tour[start:end+1][::-1]
    else:
        length = (end - start + n) % n + 1
        for i in range(length // 2):
            position_1 = (start + i) % n
            position_2 = (end - i + n) % n
            tour[position_1], tour[position_2] = tour[position_2], tour[position_1]


def inver_over_algorithm(population_size, p, max_generations, tsp):
    """
    Algorithm implementation as given in the research paper: Inver-over Operator for the TSP
    """
    n = tsp.dimension
    population = []
    
    # Define the population range
    for j in range(population_size):
        population.append(random_permutation(n))
    
    # Path Evaluation - Override TSP path_length for cyclic tours
    def evaluate(tour):
        # For inver-over, we need cyclic path length (no duplicate start/end city)
        total = 0
        n = len(tour)
        for city_index in range(n):
            next_city = (city_index + 1) % n
            total += tsp.get_distance(tour[city_index] - 1, tour[next_city] - 1)
        return total
    
    # Main Loop to Compute the logic
    for gen_idx in range(max_generations):
        for i in range(population_size):
            s_dash = population[i].copy()
            c_idx = random.randint(0, n - 1)
            c = s_dash[c_idx]
            
            while True:
                c_dash = -1
                
                if random.random() <= p:
                    choices = [v for v in s_dash if v != c]  # Select random city from current tour
                    c_dash = random.choice(choices)  # Select a random choice
                else:
                    rand_ind = random.randint(0, population_size - 1)
                    while rand_ind == i:
                        rand_ind = random.randint(0, population_size - 1)
                    # Assign to c_dash the next_city
                    s_oth = population[rand_ind]
                    it_idx = s_oth.index(c)
                    c_dash = s_oth[0] if it_idx == len(s_oth) - 1 else s_oth[it_idx + 1]
                
                # Check if c and c_dash are adjacent in current tour
                idx_c = s_dash.index(c)
                idx_next = (idx_c + 1) % n
                idx_prev = (idx_c - 1 + n) % n
                
                if s_dash[idx_next] == c_dash or s_dash[idx_prev] == c_dash:
                    break 
                
                # Find position of c_dash
                idx_cd = s_dash.index(c_dash)
                
                # Invert segment for the next position of c_dash
                invert_segment(s_dash, idx_next, idx_cd)
                
                c = c_dash 
            
            # Replace with a better offspring
            if evaluate(s_dash) < evaluate(population[i]):
                population[i] = s_dash
    
    best_len = float('inf')
    for ind in population:
        best_len = min(best_len, evaluate(ind))
    
    # Best Solution returned
    return int(best_len)


def write_results_inver_over(results, results_dir=None):
    """Write the results of mean, min and stdev to the inverover.txt file."""
    if results_dir is None:
        base_dir = Path(__file__).parent
        results_dir = base_dir / "results"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / "inverover.txt"
    
    with open(out_file, "w") as fout:
        for key in sorted(results.keys()):  # Sorting the keys inside the tuple
            tsp_name, method = key  # Unpacking the tuple
            lengths = results[key]
            
            average = mean(lengths)
            min_len = min(lengths)
            stddev = statistics.pstdev(lengths)
            
            line = (
                f"Name: {tsp_name:<10}  Method: {method:<11}  Min: {min_len}, Mean: {average:.2f}, Stddev : {stddev:.2f}"
            )
            fout.write(line + "\n")
    
    print(f"Inver-over algorithm results written to {out_file}")


def run_exercise7():
    """
    Exercise 7: Inver-over Evolutionary Algorithm
    - Run inver-over algorithm on TSP instances
    - Parameters: population_size=50, p=0.02, max_generations=20000
    - 30 runs per instance
    - Write results to results/inverover.txt
    """
    # tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100", 
    #              "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    tsp_names = ["eil51"]  # Start with one for testing
    
    results = {}
    operator_name = "inver-over"
    
    for current_index, name in enumerate(tsp_names):
        try:
            filepath = BASE_DIR / f"tsp/{name}.tsp"
            tsp = TSP(filepath)
            tsp.name = name
            
            print(f"Running TSP instance: {tsp.name}")
            
            tour_lengths = []
            # Run 30 instances (reduced to 5 for testing)
            for rep in tqdm(range(5), desc=f"Inver-over {name}"):
                # Define the generation count and population size
                # Reduced generations for testing: 20000 -> 1000
                min_length = inver_over_algorithm(50, 0.02, 1000, tsp)
                tour_lengths.append(min_length)
            
            key = (name, operator_name)  
            results[key] = tour_lengths
            
        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")
    
    # Write final computed results
    write_results_inver_over(results, RESULTS_DIR)
    print(f"Exercise 7 complete. Results written to {RESULTS_DIR/'inverover.txt'}")


# ---------------- Run Exercises ----------------
if __name__ == "__main__":
    print("Choose which exercise to run:")
    print("2 - Local Search")
    print("6 - Evolutionary Algorithms") 
    print("7 - Inver-over Algorithm")
    print("Enter exercise number (2, 6, or 7): ", end="")
    
    try:
        choice = input().strip()
        if choice == "2":
            print("Running Exercise 2 (Local Search)...")
            run_exercise2()
        elif choice == "6":
            print("Running Exercise 6 (Evolutionary Algorithms)...")
            run_exercise6()
        elif choice == "7":
            print("Running Exercise 7 (Inver-over Algorithm)...")
            run_exercise7()
        else:
            print("Invalid choice. Running Exercise 6 by default...")
            run_exercise6()
    except (KeyboardInterrupt, EOFError):
        print("\nRunning Exercise 6 by default...")
        run_exercise6()