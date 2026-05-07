import numpy as np
import pandas as pd
import random
from walkforward import *
###################################################### requirements for GA:#############################################################

# define range for each gene
def generate_gene(): 
    # Generate the test period (between 10 and 200 days, in steps of 10)
    test_period = random.choice(range(10, 201, 10))
    #test_period = random.choice(range(20, 100, 5))
    
    # Generate the train period (between 10 and 200 days, in steps of 10) such that train_period >= test_period
    train_period = random.choice(range(test_period, 201, 10))  # Train period must be >= test period
    #train_period = random.choice(range(test_period, 100, 5))  # Train period must be >= test period
    
    # Return the chromosome as a tuple (train_period, test_period)
    return np.array([train_period, test_period])


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

def selection(population,data,entire_params_dict, allPossibleWeights, weight_return=0.8, weight_risk=0.2):
    parents = []
    parents_dict = {}
    sorted_parents = {}
    for i in range(len(population)):

        if tuple(population[i][0]) in entire_params_dict:
            cost = entire_params_dict[tuple(population[i][0])]        
        else:    
            cost = walkForwardOptimization3(data, population[i][0], allPossibleWeights, create_csv=False, weight_return=weight_return, weight_risk=weight_risk)
            entire_params_dict[tuple(population[i][0])] = cost

        parents_dict[tuple(population[i][0])] = cost
    sorted_parents = sorted(parents_dict.items(), key=lambda x: x[1], reverse= True)
    selected_parents = sorted_parents[:round(len(sorted_parents)*0.75)] 

    for i in selected_parents:
        parents.append(np.array(i[0]))

    return parents, selected_parents, entire_params_dict


def crossover(selected_parents,data,entire_params_dict, allPossibleWeights, weight_return=0.8, weight_risk=0.2):
    offsprings = []
    offsprings_dict = {}
    sorted_offsprings = {}

    for _ in range(round(len(selected_parents)//1)):
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

        if (offspring_0[0] >= offspring_0[1]):
            offsprings.append(offspring_0)
        if (offspring_1[0] >= offspring_1[1]):
            offsprings.append(offspring_1)


    for i in range(len(offsprings)):
        if tuple(offsprings[i]) in entire_params_dict:
            cost = entire_params_dict[tuple(offsprings[i])]
        else:
            cost = walkForwardOptimization3(data, offsprings[i], allPossibleWeights, create_csv=False, weight_return=weight_return, weight_risk=weight_risk)
            entire_params_dict[tuple(offsprings[i])] = cost

        offsprings_dict[tuple(offsprings[i])] = cost
    sorted_offsprings = sorted(offsprings_dict.items(), key=lambda x: x[1], reverse= True)
    selected_offsprings = sorted_offsprings
   
    return offsprings, selected_offsprings, entire_params_dict


def mutation(individuals, mutation_rate,data,entire_params_dict, allPossibleWeights, weight_return=0.8, weight_risk=0.2):
    mutated_individuals = []
    mutated_dict = {}
    sorted_mutated = {}
    for i in range(len(individuals)):
        for g in range(len(individuals[i])):
            if random.random() < mutation_rate:
                if g == 0:
                    individuals[i][g] = random.choice(range(int(individuals[i][1]), 201, 10))
                    #individuals[i][g] = random.choice(range(int(individuals[i][1]), 100, 5))
                elif g == 1 and individuals[i][0] != 10:
                    individuals[i][g] = random.choice(range(10, int(individuals[i][0]), 10))
                    #individuals[i][g] = random.choice(range(20, int(individuals[i][0]), 5))

        if (individuals[i][0] >= individuals[i][1]):
            mutated_individuals.append(individuals[i])


    for i in range(len(mutated_individuals)):
        if tuple(mutated_individuals[i]) in entire_params_dict:
            cost = entire_params_dict[tuple(mutated_individuals[i])]
        else:
            cost = walkForwardOptimization3(data, mutated_individuals[i], allPossibleWeights, create_csv=False, weight_return=weight_return, weight_risk=weight_risk)
            entire_params_dict[tuple(mutated_individuals[i])] = cost
        mutated_dict[tuple(mutated_individuals[i])] = cost
    sorted_mutated = sorted(mutated_dict.items(), key=lambda x: x[1], reverse= True)
    selected_mutated = sorted_mutated
        
    return mutated_individuals, selected_mutated, entire_params_dict


def GA2(population_size, generations, data, possibleWeights, weight_return=0.8, weight_risk=0.2):
    generation_dict = {"generation":[], "best_param":[], "best_cost":[]}
    entire_params_dict = {}
    population = create_population(population_size)
    mutation_rate = initial_mutation_rate
    for g in range(generations):
        # Selection
        parents, selected_parents, entire_params_dict = selection(population, data, entire_params_dict, possibleWeights, weight_return, weight_risk)

        # Crossover
        offsprings, selected_offsprings, entire_params_dict = crossover(selected_parents,data, entire_params_dict, possibleWeights, weight_return, weight_risk)

        # Mutation        
        _, selected_mutated, entire_params_dict = mutation(offsprings + parents, mutation_rate,data, entire_params_dict, possibleWeights, weight_return, weight_risk)


        population = selected_parents + selected_offsprings + selected_mutated
        population = list(set(population))
        sorted_population = sorted(population, key=lambda x:x[1], reverse=True)
        population = sorted_population[:population_size]

        generation_dict["generation"].append(g)
        generation_dict["best_param"].append(sorted_population[0][0])
        generation_dict["best_cost"].append(sorted_population[0][1])

        print(f"*** GA2 *** : generation {g} : parameters = {sorted_population[0][0]} , cost = {sorted_population[0][1]}")

        if g >= 15:
            if (mutation_rate == initial_mutation_rate) and (all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])):
                mutation_rate = high_mutation_rate
                probable_optimal_gen = g
            elif (mutation_rate == high_mutation_rate) and (not all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])):
                mutation_rate = initial_mutation_rate
            elif (mutation_rate == high_mutation_rate) and (all(x >= generation_dict["best_cost"][-1] for x in generation_dict["best_cost"][-8:-1])) and (g > probable_optimal_gen + 1):
                break


    return generation_dict


