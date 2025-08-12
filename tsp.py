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
        for (tsp_name, method), (min_len, avg_len) in results.items():
            f.write(
                f"Name: {tsp_name}  Method: {method}   Min: {min_len:.2f}, Mean: {avg_len:.2f}\n")
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


# The function that connects operator with relative functions
def local_search(tsp: TSP, neighbourhood_operator):
    # Creating an initial solution with a random permutation of cities
    current = random_permutation(tsp.dimension)

    if neighbourhood_operator == "jump":
        _, local_min_length = get_jump_local_minimum(tsp, current, float('inf'))
    elif neighbourhood_operator == "exchange":
        _, local_min_length = get_exchange_local_minimum(tsp, current, float('inf'))
    elif neighbourhood_operator == "2opt":
        _, local_min_length = get_2opt_local_minimum(tsp, current, float('inf'))
    else:
        raise ValueError("Unknown neighbourhood operator")
    

    # print(f"Minimum neighbour length: {local_min}")
    # print(f"Average neighbour length: {avg_length:.2f}")

    return local_min_length


# -------------------------------
# Best Neighbour Generator
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
    # return the tour and the current tour's length
    return current, current_path_length

# Just copy and pasted the jump function, however, just changed sum iteration logic of indices 'i' and 'j' 
# to iterate as exchange and 2opt needs
def get_exchange_local_minimum(tsp, tour, max_iterations):
    """Generate all neighbours by swapping two cities."""
    iterations = 0
    current = tour[:]
    current_path_length = tsp.path_length(tour)

    while iterations < max_iterations:
        iterations += 1
        better_neighbour_found = False
        n = len(current)

        # need to change the logic here a little bit, searching for every exchange possibility
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
                new_tour_body = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
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



# # decided to have a separate function to generate all neighbours from the local minimum to keep it more modular
# # but after changing up the code for efficiency, it is not needed at all for now.
# def get_neighbourhood(tour):
#     tour_body = tour[1:-1]
#     n = len(tour_body)
#     neighbourhood = []
#     for i in range(n): 
#         for j in range(n):
#             if i == j:
#                 continue
#             new_tour_body = tour_body[:]
#             city = new_tour_body.pop(i)
#             new_tour_body.insert(j, city)
#             neighbourhood.append([tour[0]] + new_tour_body + [tour[-1]])
#     return neighbourhood


def random_permutation(n):
    perm = list(range(2, n + 1))
    random.shuffle(perm)
    perm = [1] + perm + [1]
    return perm


# Function use to run local_search that conencts to relevant neighbour generators based on operator inputted.
# The understanding is that each instance, keep iterating until no better neighbour can be found.

# Once every solution of 30 instances of a file i.e. "eil51" of one operator i.e. "jump" is found,
# The average and minimum score is found for all of these instances and outputting into the results folder. 
def run_all_instances():
    tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100",
                "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    results = {}

    for name in tsp_names:
        try:
            filepath = Path(f"tsp/{name}.tsp")
            tsp = TSP(filepath)
            tsp.name = name
            print(f"Running TSP instance: {tsp.name}")

            # for operator_name in {"jump","exchange","2opt"}:
            for operator_name in {"jump", "exchange", "2pot"}:
                average = 0
                minimum =float('inf')
                for _ in tqdm(range(30), desc=f"{name} - {operator_name}"):
                    min_length = local_search(tsp, operator_name)
                    average += min_length
                    if min_length < minimum:
                        minimum = min_length
                average /= 30
                results[(name, operator_name)] = (minimum, average)

        except Exception as e:
            print(f"Failed loading or processing {name}: {e}")

    write_results(results)

def main():
    run_all_instances()


if __name__ == "__main__":
    main()
