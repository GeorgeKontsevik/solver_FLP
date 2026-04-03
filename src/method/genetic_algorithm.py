import pandas as pd
import random
import pulp
import numpy as np
from tqdm import trange
from tqdm import tqdm
from .location_problem import (
    solve_combined_problem,
    resolve_demand_series,
    resolve_existing_facility_mask,
)

# Generate an initial population with random numbers
def generate_population(res, accessibility_matrix_demand, population_size, number_res):
    
    matrix_range = (0.5, 0.9)  
    new_matrix = accessibility_matrix_demand.copy()
    pops = []
    res_selected = []

    if number_res == 'all':
        res_selected = res
    else:
        for _ in range(number_res):
            value = random.choice(res)
            res.remove(value)
            res_selected.append(value)

    for _ in range(population_size):
        for i in res_selected:
            new_matrix.loc[i[0], i[1]] = random.uniform(matrix_range[0], matrix_range[1])*accessibility_matrix_demand.loc[i[0], i[1]]
            new_matrix.loc[i[1], i[0]] = new_matrix.loc[i[0], i[1]]
        pops.append(new_matrix)

    return pops, res

# Perform selection based on fitness scores
def selection(population, num_parents, df, service_radius, my_version):
    sorted_population = sorted(population, key=lambda x: calculate_fitness(x, df, service_radius, my_version), reverse=False)
    parents = sorted_population[:num_parents]
    return parents

# Perform crossover to create offspring
def crossover(parents, num_offspring, matrix):

    matrix_size = len(matrix)

    offspring = []
    while len(offspring) < num_offspring:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        crossover_point = random.randint(1, matrix_size - 1)
        child1 = pd.DataFrame(np.vstack((parent1[:crossover_point], parent2[crossover_point:])))
        child2 = pd.DataFrame(np.vstack((parent2[:crossover_point], parent1[crossover_point:])))

        offspring.append(child1)
        offspring.append(child2)

    return offspring

# Perform mutation to introduce random changes
def mutation(offspring, res, mutation_rate):

    matrix_range = (0.5, 9.0)
    offspring_mutationed = []

    for child in offspring:
        if random.random() < mutation_rate:
            index = random.choice(res)
            child.loc[index[0], index[1]] = random.uniform(matrix_range[0], matrix_range[1]) * child.loc[index[0], index[1]]
            child.loc[index[1], index[0]] = child.loc[index[0], index[1]]
            
        offspring_mutationed.append(child)
    
    return offspring_mutationed
    
def calculate_fitness(
    candidate_matrix,
    df,
    service_radius,
    demand_column=None,
    prefer_existing=False,
    existing_facility_discount=1.0,
    existing_column=None,
):
    try:
        demand_quantity = resolve_demand_series(df, demand_column=demand_column)
        existing_facility_mask = resolve_existing_facility_mask(df, existing_column=existing_column)
        facilities, capacities, fac2cli = solve_combined_problem(np.array(candidate_matrix),
                                                        service_radius,
                                                        demand_quantity,
                                                        existing_facility_mask=existing_facility_mask,
                                                        prefer_existing=prefer_existing,
                                                        existing_facility_discount=existing_facility_discount)
        dict_fitness = dict([(k,l) for k,l in enumerate(fac2cli) if len(l)>0])

        fitness = len(dict_fitness)
        if prefer_existing:
            for key, _ in dict_fitness.items():
                if existing_facility_mask is not None and existing_facility_mask[key]:
                    fitness -= float(existing_facility_discount)

        return fitness
    except RuntimeError:
        return 0
    
    # Main genetic algorithm
def genetic_algorithm_main(matrix, edges, population_size, num_generations, df, 
                      service_radius, mutation_rate, num_parents, num_offspring, number_res,
                      demand_column=None, prefer_existing=False,
                      existing_facility_discount=1.0, existing_column=None):
    
    population, res_ost = generate_population(edges, matrix, population_size, number_res)
    fitness_history = []  # История изменения фитнеса

    for generation in trange(num_generations):
        # Рассчитываем фитнес и отсортированную популяцию
        population_with_fitness = [
            (
                individual,
                calculate_fitness(
                    individual,
                    df,
                    service_radius,
                    demand_column=demand_column,
                    prefer_existing=prefer_existing,
                    existing_facility_discount=existing_facility_discount,
                    existing_column=existing_column,
                ),
            )
            for individual in population
        ]
        
        # Сохраняем фитнес текущей популяции
        fitness_history.append([fitness for _, fitness in population_with_fitness])

        # Отбираем родителей
        parents = [individual for individual, _ in sorted(population_with_fitness, key=lambda x: x[1])[:num_parents]]

        # Генерируем потомков
        offspring = crossover(parents, num_offspring, matrix)

        # Применяем мутацию
        offspring_mutationed = mutation(offspring, res_ost, mutation_rate)

        # Обновляем популяцию
        population = parents + offspring_mutationed

    # Получаем лучшее решение
    best_candidate, _ = min(population_with_fitness, key=lambda x: x[1])

    return best_candidate, fitness_history

def choose_edges(sim_matrix, service_radius):

    edges = []

    for i in tqdm(sim_matrix.index):
        for j in sim_matrix.columns:
            if sim_matrix.loc[i, j] >= service_radius and i != j:
                # Reduce by 40% if the value is 15 or greater
                variant = sim_matrix.copy()
                if variant.loc[i, j] > service_radius and variant.loc[i, j]*0.6 <= service_radius:
                    if [j, i] not in edges:
                        edges.append([i,j])

    return edges
