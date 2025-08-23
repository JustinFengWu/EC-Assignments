# Exercise 7 - Inver-over Evolutionary Algorithm

# dependencies
import os
import math
import random
import statistics
from tsp import TSP as Original_TSP

#Override the original TSP class defined in tsp.py to adhere to code modularity and avoid repetition. 
class TSP(Original_TSP):
    def path_length(self, perm):
        total = 0
        n = len(perm)
        for city_index in range(n):
            next_city = (city_index + 1) % n
            total += self.get_distance(perm[city_index] - 1, perm[next_city] - 1)
        return total

#This function writes the results of mean, min and stdev to the inverover.txt file
def write_results_inver_over(results):
    os.makedirs("results", exist_ok = True)
    
    with open("results/inverover.txt", "w") as fout:

        for key in sorted(results.keys()): #Sorting the keys inside the tuple
            tsp_name, method = key  # Unpacking the tuple
            lengths = results[key]
            
            average = mean(lengths)
            min_len = min(lengths)
            stddev = statistics.pstdev(lengths)
            
            line = (
                f"Name: {tsp_name:<10}  Method: {method:<11}  Min: {min_len}, Mean: {average:.2f}, Stddev : {stddev:.2f}"
            )
            fout.write(line + "\n")
    
    print("Inver-over algorithm results written to results/inverover.txt")

#Mean function to calculate the average of all values and store in inverover.txt file
def mean(vals):
    return sum(vals) / len(vals)

#Stdev function to calculate the standard deviation using integer arithmetic. 
def stdev(vals):
    if len(vals) <= 1:
        return 0.0
    mean_value = mean(vals)
    accumulator = sum((x - mean_value) ** 2 for x in vals)
    return math.sqrt(accumulator / (len(vals) - 1))

#Generating random permutation
def random_permutation(n):
    permutation = list(range(1, n + 1))
    random.shuffle(permutation)
    return permutation

#Funnction to invert the halft segment of the array
def invert_segment(tour, start, end):
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

#Algorithm implementation as given in the research paper :- Inver-over Operator for the TSP
def inver_over_algorithm(population_size, p, max_generations, tsp):
    n = tsp.dimension
    population = []
    
    #Defining the population range
    for j in range(population_size):
        population.append(random_permutation(n))
    
    #Path Evaluation
    def evaluate(tour):
        return tsp.path_length(tour)
    
    # Main Loop to Compute the logic
    for gen_idx in range(max_generations):

        for i in range(population_size):
            s_dash = population[i].copy()
            c_idx = random.randint(0, n - 1)
            c = s_dash[c_idx]
            
            while True:
                c_dash = -1
                
                if random.random() <= p:
                    choices = [v for v in s_dash if v != c] #Selecting the random city from the current tour
                    c_dash = random.choice(choices) #Select a random choice
                else:
                    rand_ind = random.randint(0, population_size - 1)
                    while rand_ind == i:
                        rand_ind = random.randint(0, population_size - 1)
                    #Assigning to c_dash the next_cit
                    s_oth = population[rand_ind]
                    it_idx = s_oth.index(c)
                    c_dash = s_oth[0] if it_idx == len(s_oth) - 1 else s_oth[it_idx + 1]
                
                #Checking if c and c_dash are adjacent in current tour
                idx_c = s_dash.index(c)
                idx_next = (idx_c + 1) % n
                idx_prev = (idx_c - 1 + n) % n
                
                if s_dash[idx_next] == c_dash or s_dash[idx_prev] == c_dash:
                    break 
                
                #Find position of c_dash
                idx_cd = s_dash.index(c_dash)
                
                #Invert segment for the next position of c_dash
                invert_segment(s_dash, idx_next, idx_cd)
                
                c = c_dash 
            
            #Replace with a better offspring
            if evaluate(s_dash) < evaluate(population[i]):
                population[i] = s_dash
    
    best_len = float('inf')
    for ind in population:
        best_len = min(best_len, evaluate(ind))
    
    #Best Solution returned
    return int(best_len)

def run_all_instances():
    #Running the method on all instances
    tsp_names = ["eil51", "eil76", "eil101", "kroA100", "kroC100", "kroD100", "lin105", "pcb442", "pr2392", "st70", "usa13509"]
    results = {}
    operator_name = "inver-over"
    
    os.makedirs("results", exist_ok = True)
    
    for current_index, name in enumerate(tsp_names):

        filepath = f"tsp/{name}.tsp"
        tsp = TSP(filepath)
        tsp.name = name
        
        print(f"Running TSP instance: {tsp.name} ")
        
        tour_lengths = []
        #Running on 30 instances
        for rep in range(30):
            #Defining the Generation count and the population size
            min_length = inver_over_algorithm(50, 0.02, 20000, tsp)
            tour_lengths.append(min_length)
        
        key = (name, operator_name)  
        results[key] = tour_lengths
    
    #Writing final computed results
    write_results_inver_over(results)

if __name__ == "__main__":
    run_all_instances()