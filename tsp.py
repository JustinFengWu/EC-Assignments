"""
TSP problem parser and utilities
"""

# dependencies
import math
import sys
import random
import os
from pathlib import Path


def write_results(results):
    os.makedirs("results", exist_ok=True)
    with open("results/local_search.txt", "w") as f:
        for (tsp_name, method), (min_len, avg_len) in results.items():
            f.write(
                f"{tsp_name} {method} min: {min_len:.2f}, mean: {avg_len:.2f}\n")
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
    current = random_permutation(tsp.dimension)
    current_length = tsp.path_length(current)

    if neighbourhood_operator == "jump":
        neighbourhood = get_all_jump_neighbours(current)
    elif neighbourhood_operator == "exchange":
        neighbourhood = get_all_exchange_neighbours(current)
    elif neighbourhood_operator == "2opt":
        neighbourhood = get_all_2opt_neighbours(current)
    else:
        raise ValueError("Unknown neighbourhood operator")
    print("Current: ")
    print(current)
    print("Neighbourhood: ")
    for _ in neighbourhood:
        print(f"{_}\n")

    neighbourhood = list({tuple(tour) for tour in neighbourhood})

    # Compute lengths of all neighbours
    lengths = [tsp.path_length(tour) for tour in neighbourhood]

    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths)

    print(f"Initial tour length: {current_length}")
    print(f"Minimum neighbour length: {min_length}")
    print(f"Average neighbour length: {avg_length:.2f}")

    return min_length, avg_length


# -------------------------------
# Neighbourhood generators
# -------------------------------
def get_all_jump_neighbours(tour):
    """Generate all neighbours by moving one city to a new position."""
    neighbours = []
    n = len(tour)

    # Assume closed tour: start == end → do not move first or last city
    for i in range(1, n - 1):  # Don't move the starting/ending city
        for j in range(1, n - 1):
            if i == j:
                continue
            new_tour = tour[:]
            city = new_tour.pop(i)
            new_tour.insert(j, city)
            neighbours.append(new_tour)

    return neighbours


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
    tsp_names = ["6"]
    # tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
    #              "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    results = {}

    for name in tsp_names:
        try:
            filepath = Path(f"tsp/{name}.tsp")
            tsp = TSP(filepath)
            tsp.name = name
            print(f"Running TSP instance: {tsp.name}")

            for operator_name in {"jump","exchange","2opt"}:
                for _ in range(1):
                    min_length, average_length = local_search(tsp, operator_name)
                    print(f"Name: {name}, Instance: {_}, Min: {min_length}, Mean: {average_length}")

        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")

    write_results(results)

def main():
    run_all_instances()


if __name__ == "__main__":
    main()
