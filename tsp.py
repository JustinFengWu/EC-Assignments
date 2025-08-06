"""
TSP problem parser and utilities
"""

# dependencies
import math
import sys
import random
import os
from pathlib import Path
# Adding this just because I like the loading bar
from tqdm import tqdm


def write_results(results):
    os.makedirs("results", exist_ok=True)
    with open("results/local_search_prev.txt", "a") as f:
        for (tsp_name, method, _), (min_len, avg_len) in results.items():
            f.write(
                f"Name: {tsp_name}  Method: {method}    Instance: {_ + 1}   Min: {min_len:.2f}, Mean: {avg_len:.2f}\n")
    print("Local search results written to results/local_search.txt")


class TSP:
    def __init__(self, filename):
        self.dimension = 0
        self.coordinates = []
        self.distance_matrix = []
        self.node_section = False
        self.name = ""

        self.load_tsp(filename)
        self.create_distance_matrix()

    def load_tsp(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith("DIMENSION"):
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
        total = 0
        for i in range(len(perm) - 1):
            total += self.get_distance(perm[i] - 1, perm[i + 1] - 1)
        return total

    def get_distance(self, i, j):
        return self.distance_matrix[i][j]


def local_search(tsp: TSP, neighbourhood_operator):
    # Creating an initial solution with a random permutation of cities
    current = random_permutation(tsp.dimension)

    if neighbourhood_operator == "jump":
        local_min_tour, local_min_length = get_jump_local_minimum(tsp, current, float('inf'))
        neighbourhood = get_jump_neighbourhood(local_min_tour)
    elif neighbourhood_operator == "exchange":
        local_min_tour, local_min_length = get_all_exchange_neighbours(tsp, current)
    elif neighbourhood_operator == "2opt":
        local_min_tour, local_min_length = get_all_2opt_neighbours(tsp, current)
    else:
        raise ValueError("Unknown neighbourhood operator")
    
    # remove duplicates by putting into set and converting back to list
    neighbourhood = list({tuple(tour) for tour in neighbourhood})
    # convert each neighbour into their respective path lengths
    path_lengths = [tsp.path_length(tour) for tour in neighbourhood]
    avg_length = sum(path_lengths) / len(path_lengths)

    # print(f"Minimum neighbour length: {local_min}")
    # print(f"Average neighbour length: {avg_length:.2f}")

    return local_min_length, avg_length


# -------------------------------
# Neighbourhood generators
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
        # It feels much easier to just append start and end onto the body after.
        tour_body = current[1:-1] # [1:-1] means starting at the index 1 and ending before last index
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

    # The plan here is to find the tour that is the local min, and then we could use another function to find all other neighbours
    return current, current_path_length

# decided to have a separate function to generate all neighbours from the local minimum to keep it more modular
def get_jump_neighbourhood(tour):
    tour_body = tour[1:-1]
    n = len(tour_body)
    neighbourhood = []
    for i in range(n): 
        for j in range(n):
            if i == j:
                continue
            new_tour_body = tour_body[:]
            city = new_tour_body.pop(i)
            new_tour_body.insert(j, city)
            neighbourhood.append([tour[0]] + new_tour_body + [tour[-1]])
    return neighbourhood

def get_all_exchange_neighbours(tour):
    """Generate all neighbours by swapping two cities."""
    neighbours = []
    n = len(tour)
    for i in range(1, n - 1):
        for j in range(i + 1, n - 1):
            new_tour = tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            neighbours.append(new_tour)
    return neighbours


def get_all_2opt_neighbours(tour):
    """Generate all neighbours by 2-opt reversal."""
    neighbours = []
    n = len(tour)
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
            neighbours.append(new_tour)
    return neighbours


def random_permutation(n):
    perm = list(range(2, n + 1))
    random.shuffle(perm)
    perm = [1] + perm + [1]
    return perm


def run_all_instances():
    tsp_names = ["eil51"]
    # tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
    #              "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    results = {}

    for name in tsp_names:
        try:
            filepath = Path(f"tsp/{name}.tsp")
            tsp = TSP(filepath)
            tsp.name = name
            print(f"Running TSP instance: {tsp.name}")

            # for operator_name in {"jump","exchange","2opt"}:
            for operator_name in {"jump"}:
                for _ in tqdm(range(30), desc=f"{name} - {operator_name}"):
                    min_length, average_length = local_search(tsp, operator_name)
                    # print(f"Name: {name}, Instance: {_}, Min: {min_length}, Mean: {average_length}")
                    results[(name, operator_name, _)] = (min_length, average_length)

        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")

    write_results(results)

def main():
    run_all_instances()


if __name__ == "__main__":
    main()
