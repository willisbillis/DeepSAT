from itertools import product
from random import sample, shuffle, choice
import numpy as np

PARAMS_DICT = {
    "Embedding_dim": [16, 32, 64],
    "LSTM_layers": [1, 2, 3, 4, 5],
    "LSTM_neurons": [32, 64, 128, 256, 512],
    "dense_layers": [1, 2, 3],
    "dense_neurons": [4, 8, 16, 32, 64],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5]
}

## CONSTANTS
POP_SIZE = 20
GENERATIONS = 15
ELITISM_RATE = 0.1
MUTATION_RATE = 0.05

def fitness_function(member):
    value = 0
    weight = 0
    if member["Size"] == "small":
        weight = 1 * member["Counts"]
    elif member["Size"] == "medium":
        weight = 2 * member["Counts"]
    elif member["Size"] == "large":
        weight = 3 * member["Counts"]
    if member["Color"] == "red":
        value = 7 * member["Counts"]
    elif member["Color"] == "orange":
        value = 5 * member["Counts"]
    elif member["Color"] == "yellow":
        value = 3 * member["Counts"]
    if weight > 8:
        value = 0
    return value

class Generation:
    """ Generation class for population """
    def __init__(self, params_dict, pop_size):
        self.gen_count = 1
        self.params_dict = params_dict
        self.params = list(params_dict.keys())
        tot_param_set = list(product(*params_dict.values()))
        population_params = sample(tot_param_set, pop_size)
        members = []
        for member_id, params in enumerate(population_params):
            new_member = {
                "ID": member_id,
                }
            for param_name, param_val in zip(self.params, params):
                new_member[param_name] = param_val
            members.append(new_member)
        self.population = members

    def evaluate_fitness(self, fitness_func):
        for child in self.population:
            if "Fitness" not in child.keys():
                child["Fitness"] = fitness_func(child)
        fitness_list = [child["Fitness"] for child in self.population]
        avg_fitness = np.mean(fitness_list)
        peak_fitness = max(fitness_list)
        return avg_fitness, peak_fitness

    def next_generation(self, elitism_rate, mutation_rate):
        new_generation = []
        new_parents = []
        old_generation = sorted(self.population, key=lambda child: child["Fitness"])
        # Step 1: Elitism
        elitism_count = round(len(old_generation) * elitism_rate)
        new_parents.extend(old_generation[-elitism_count:])
        runt_count = round(elitism_count/2.0)
        old_generation = old_generation[runt_count:-elitism_count]
        # Step 2: Tournament style natural selection
        shuffle(old_generation)
        while len(old_generation) > 1:
            member_1 = old_generation.pop()
            member_2 = old_generation.pop()
            best_fitness = max(member_1["Fitness"], member_2["Fitness"])
            for member in (member_1, member_2):
                if member["Fitness"] == best_fitness:
                    new_parents.append(member)
        if len(old_generation) == 1:
            new_parents.append(old_generation[0])
        # Step 3: Cross replication
        shuffle(new_parents)
        while len(new_parents) > 1:
            parent_1 = new_parents.pop()
            parent_2 = new_parents.pop()
            new_child_1 = {"ID": parent_1["ID"] + len(self.population)}
            new_child_2 = {"ID": parent_2["ID"] + len(self.population)}
            for idx, param in enumerate(self.params):
                if idx % 2 == 0:
                    new_child_1[param] = parent_1[param]
                    new_child_2[param] = parent_2[param]
                else:
                    new_child_1[param] = parent_2[param]
                    new_child_2[param] = parent_1[param]
            new_generation.extend((parent_1, parent_2, new_child_1, new_child_2))
        if len(new_parents) == 1:
            new_generation.append(new_parents[0])
        # Step 4: Mutation
        shuffle(new_generation)
        mutation_count = round(len(new_generation) * mutation_rate)
        mutated = new_generation[-mutation_count:]
        new_generation = new_generation[:-mutation_count]
        for mutant in mutated:
            mutated_key = choice(self.params)
            mutant[mutated_key] = choice(self.params_dict[mutated_key])
        new_generation += mutated
        # Step 5: set object variables
        self.population = new_generation
        self.gen_count += 1

G1 = Generation(PARAMS_DICT, POP_SIZE)
print(G1.evaluate_fitness(fitness_function))

while G1.gen_count < GENERATIONS:
    G1.next_generation(ELITISM_RATE, MUTATION_RATE)
    print(G1.evaluate_fitness(fitness_function))
    print(len(G1.population))
