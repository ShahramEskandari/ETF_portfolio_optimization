import numpy as np
import pandas as pd
import random
from fitness_ga1 import *
from makeArrayOfWeights import is_valid
###################################################### requirements for GA:#############################################################

# define range for each gene
def generate_gene():
    while True:
        # Generate random portfolio weights
        gene = np.random.rand(8)
        
        # Normalize the gene so that the sum equals 1
        gene = gene / np.sum(gene)
        
        # Round the weights to 3 decimal places
        gene = np.round(gene, 3)
        
        # Adjust the last weight to ensure the sum is exactly 1
        gene[7] = 1 - np.sum(gene[:7])
        
        # Ensure no weight is negative (and check conditions)
        if (gene[2] + gene[3] + gene[4] >= 0.125 and
            gene[5] + gene[6] + gene[7] >= 0.125 and
            gene[0] + gene[1] >= 0.125 and
            np.all(gene >= 0)):  # Ensure all weights are >= 0
            return gene


initial_mutation_rate = 0.05
high_mutation_rate = 0.1
########################################################################################################################################

def create_population(population_size):
    population = []
    for _ in range(population_size):
        individual = generate_gene()   
        population.append(individual)
    
    for i in range(len(population)):
        population[i] = (population[i],)

    return population

def selection(population,data,entire_params_dict, extermums, weight_return=0.8, weight_risk=0.2):
    parents = []
    parents_dict = {}
    sorted_parents = {}
    for i in range(len(population)):

        if tuple(population[i][0]) in entire_params_dict:
            cost = entire_params_dict[tuple(population[i][0])]        
        else:    
            cost = fitness_function(data,population[i][0], extermums, weight_return, weight_risk)
            entire_params_dict[tuple(population[i][0])] = cost

        parents_dict[tuple(population[i][0])] = cost
    sorted_parents = sorted(parents_dict.items(), key=lambda x: x[1], reverse= True)
    selected_parents = sorted_parents[:round(len(sorted_parents)*0.75)] 

    for i in selected_parents:
        parents.append(np.array(i[0]))

    return parents, selected_parents, entire_params_dict


def crossover(selected_parents,data,entire_params_dict, extermums, weight_return=0.8, weight_risk=0.2):
    offsprings = []
    offsprings_dict = {}
    sorted_offsprings = {}

    for _ in range(round(len(selected_parents)*1.4)):
        parents = random.sample(selected_parents,2)
        crossover_point = random.randint(1, len(parents[0][0]) - 1)
        offspring_0 = np.empty(len(parents[0][0]))
        offspring_1 = np.empty(len(parents[0][0]))
        for j in range(len(parents[0][0])):
            if j < crossover_point:
                offspring_0[j] = parents[0][0][j]
                offspring_1[j] = parents[1][0][j]
            else:
                offspring_0[j] = parents[1][0][j]
                offspring_1[j] = parents[0][0][j]
        offspring_0 = offspring_0/np.sum(offspring_0)
        offspring_0 = np.floor(offspring_0*1000)/1000
        offspring_0[-1] = round(1- np.sum(offspring_0[:-1]),3)
        offspring_1 = offspring_1/np.sum(offspring_1)
        offspring_1 = np.floor(offspring_1*1000)/1000
        offspring_1[-1] = round(1- np.sum(offspring_1[:-1]),3)

        if (is_valid(offspring_0)):
            offsprings.append(offspring_0)
        if (is_valid(offspring_1)):
            offsprings.append(offspring_1)


    for i in range(len(offsprings)):
        if tuple(offsprings[i]) in entire_params_dict:
            cost = entire_params_dict[tuple(offsprings[i])]
        else:
            cost = fitness_function(data,offsprings[i], extermums, weight_return, weight_risk)
            entire_params_dict[tuple(offsprings[i])] = cost

        offsprings_dict[tuple(offsprings[i])] = cost
    sorted_offsprings = sorted(offsprings_dict.items(), key=lambda x: x[1], reverse= True)
    selected_offsprings = sorted_offsprings
   
    return offsprings, selected_offsprings, entire_params_dict


def mutation(individuals, mutation_rate,data,entire_params_dict, extermums, weight_return=0.8, weight_risk=0.2):
    mutated_individuals = []
    mutated_dict = {}
    sorted_mutated = {}
    for i in range(len(individuals)):
        for g in range(len(individuals[i])):
            if random.random() < mutation_rate:
                individuals[i][g] = round(np.random.rand(),3)
                individuals[i] = individuals[i]/np.sum(individuals[i])
                individuals[i] = np.floor(individuals[i]*1000)/1000
                individuals[i][-1] = round(1-(np.sum(individuals[i][:-1])),3)
        if (is_valid(individuals[i])):
            mutated_individuals.append(individuals[i])


    for i in range(len(mutated_individuals)):
        if tuple(mutated_individuals[i]) in entire_params_dict:
            cost = entire_params_dict[tuple(mutated_individuals[i])]
        else:
            cost = fitness_function(data,mutated_individuals[i], extermums, weight_return, weight_risk)
            entire_params_dict[tuple(mutated_individuals[i])] = cost
        mutated_dict[tuple(mutated_individuals[i])] = cost
    sorted_mutated = sorted(mutated_dict.items(), key=lambda x: x[1], reverse= True)
    selected_mutated = sorted_mutated
        
    return mutated_individuals, selected_mutated, entire_params_dict


def GA(population_size, generations, data, possibleWeights, weight_return=0.8, weight_risk=0.2):
    ret_risk_extermums = findMinMax(data, possibleWeights)
    generation_dict = {"generation":[], "best_param":[], "best_cost":[]}
    entire_params_dict = {}
    population = create_population(population_size)
    mutation_rate = initial_mutation_rate
    for g in range(generations):
        # Selection
        parents, selected_parents, entire_params_dict = selection(population, data, entire_params_dict, ret_risk_extermums, weight_return, weight_risk)

        # Crossover
        offsprings, selected_offsprings, entire_params_dict = crossover(selected_parents,data, entire_params_dict, ret_risk_extermums, weight_return, weight_risk)

        # Mutation        
        _, selected_mutated, entire_params_dict = mutation(offsprings + parents, mutation_rate,data, entire_params_dict, ret_risk_extermums, weight_return, weight_risk)


        population = selected_parents + selected_offsprings + selected_mutated
        population = list(set(population))
        sorted_population = sorted(population, key=lambda x:x[1], reverse=True)
        population = sorted_population[:population_size]

        generation_dict["generation"].append(g)
        generation_dict["best_param"].append(sorted_population[0][0])
        generation_dict["best_cost"].append(sorted_population[0][1])

        print(f"-GA1 : generation {g} : parameters = {sorted_population[0][0]} , cost = {sorted_population[0][1]}")

        if g >= 15:
            if (mutation_rate == initial_mutation_rate) and (all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])):
                mutation_rate = high_mutation_rate
                probable_optimal_gen = g
            elif (mutation_rate == high_mutation_rate) and (not all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])):
                mutation_rate = initial_mutation_rate
            elif (mutation_rate == high_mutation_rate) and (all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])) and (g > probable_optimal_gen + 1):
                break


    return generation_dict


