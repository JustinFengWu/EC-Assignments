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


def local_search(tsp: TSP, neighbourhood_operator, max_iters=1000):
    current = random_permutation(tsp.dimension)
    current_length = tsp.path_length(current)

    for _ in range(max_iters):
        candidate = neighbourhood_operator(current.copy())
        candidate_length = tsp.path_length(candidate)
        if candidate_length < current_length:
            current, current_length = candidate, candidate_length

    return current_length


def jump(perm):
    first = random.randint(1, len(perm) - 2)
    second = first
    while second == first:
        second = random.randint(1, len(perm) - 2)

    val = perm.pop(second)
    perm.insert(first + 1, val)
    perm[-1] = perm[0]
    return perm


def exchange(perm):
    i = random.randint(1, len(perm) - 2)
    j = i
    while j == i:
        j = random.randint(1, len(perm) - 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def two_opt(perm):
    i = random.randint(1, len(perm) - 3)
    j = random.randint(i + 1, len(perm) - 2)
    perm[i:j+1] = reversed(perm[i:j+1])
    return perm


def random_permutation(n):
    perm = list(range(2, n + 1))
    random.shuffle(perm)
    perm = [1] + perm + [1]
    return perm


def run_all_instances():
    # tsp_names = ["eil51"]
    tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
                 "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    results = {}

    for name in tsp_names:
        try:
            filepath = Path(f"tsp/{name}.tsp")
            tsp = TSP(filepath)
            tsp.name = name
            print(f"Running TSP instance: {tsp.name}")

            for operator_name, operator_fn in {
                "jump": jump,
                "exchange": exchange,
                "2opt": two_opt
            }.items():
                lengths = []
                for _ in range(30):
                    length = local_search(tsp, operator_fn)
                    lengths.append(length)
                min_len = min(lengths)
                avg_len = sum(lengths) / len(lengths)
                results[(tsp.name, operator_name)] = (min_len, avg_len)

        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")

    write_results(results)


def load_all_tsps():
    from pathlib import Path
    instances = []
    tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
                 "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]

    for name in tsp_names:
        try:
            filepath = Path(f"tsp/{name}.tsp")
            tsp = TSP(filepath)
            tsp.name = name
            instances.append(tsp)
        except Exception as e:
            print(f"Failed loading {name}: {e}")
    return instances


def main():
    run_all_instances()


if __name__ == "__main__":
    main()
