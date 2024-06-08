import numpy as np
import matplotlib.pyplot as plt

# Generación de ciudades aleatorias
def generate_cities(num_cities):
    return np.random.rand(num_cities, 2) * 100

# Función de distancia entre ciudades
def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Función de fitness (distancia total del recorrido)
def calculate_fitness(route, cities):
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1)) + distance(cities[route[-1]], cities[route[0]])

# Métodos de Inicialización de Población
def random_initialization(num_individuals, num_cities):
    return [np.random.permutation(num_cities).tolist() for _ in range(num_individuals)]

def heuristic_initialization(num_individuals, num_cities, cities):
    population = []
    for _ in range(num_individuals):
        individual = []
        current_city = np.random.randint(num_cities)
        individual.append(current_city)
        remaining_cities = list(set(range(num_cities)) - {current_city})
        while remaining_cities:
            next_city = min(remaining_cities, key=lambda city: distance(cities[current_city], cities[city]))
            individual.append(next_city)
            current_city = next_city
            remaining_cities.remove(next_city)
        population.append(individual)
    return population

def hybrid_initialization(num_individuals, num_cities, cities):
    half_random = num_individuals // 2
    half_heuristic = num_individuals - half_random
    random_population = random_initialization(half_random, num_cities)
    heuristic_population = heuristic_initialization(half_heuristic, num_cities, cities)
    return random_population + heuristic_population

# Métodos de Selección
def tournament_selection(population, fitness, tournament_size=5):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(range(len(population)), size=tournament_size)
        winner = max(participants, key=lambda i: fitness[i])
        selected.append(population[winner])
    return selected

# Función de mutación
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(individual))
            individual[i], individual[j] = individual[j], individual[i]

# Función de cruce (crossover)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]
    current_pos = end + 1
    for city in parent2:
        if city not in child:
            if current_pos >= size:
                current_pos = 0
            child[current_pos] = city
            current_pos += 1
    return child

# Ejecución de un ciclo de algoritmo genético
def run_ga(population, cities, fitness_func, num_generations, mutation_rate):
    fitness_history = []
    for generation in range(num_generations):
        fitness_values = [fitness_func(individual, cities) for individual in population]
        fitness_history.append(min(fitness_values))
        selected_population = tournament_selection(population, fitness_values)
        next_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)  # Se cruza de la misma manera para simplicidad
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_population.extend([child1, child2])
        population = next_population
    return population, fitness_history

# Visualización de la ruta
def plot_route(cities, route, title):
    ordered_cities = [cities[i] for i in route]
    ordered_cities.append(ordered_cities[0])  # Retorno al inicio
    plt.figure()
    plt.plot([city[0] for city in ordered_cities], [city[1] for city in ordered_cities], marker='o')
    plt.title(title)
    plt.show()

# Visualización del fitness
def plot_fitness(generations, fitness_histories, labels):
    plt.figure()
    for fitness_history, label in zip(fitness_histories, labels):
        plt.plot(generations, fitness_history, label=label)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Comparación de Métodos de Inicialización")
    plt.show()

# Configuración de hiperparámetros
NUM_CITIES = 100
NUM_INDIVIDUALS = 200
NUM_GENERATIONS = 3000
MUTATION_RATE = 0.01

# Ejemplo de llamada principal
if __name__ == "__main__":
    cities = generate_cities(NUM_CITIES)

    # Inicialización de población
    initial_population = random_initialization(NUM_INDIVIDUALS, NUM_CITIES)

    # Ejecución del Experimento 2
    fitness_histories_2 = []
    labels_2 = ["Random", "Heuristic", "Hybrid"]
    initialization_methods = [
        lambda num_individuals, num_cities: random_initialization(num_individuals, num_cities),
        lambda num_individuals, num_cities: heuristic_initialization(num_individuals, num_cities, cities),
        lambda num_individuals, num_cities: hybrid_initialization(num_individuals, num_cities, cities)
    ]

    for init_method, label in zip(initialization_methods, labels_2):
        population = init_method(NUM_INDIVIDUALS, NUM_CITIES)
        population, fitness_history = run_ga(population, cities, calculate_fitness, NUM_GENERATIONS, MUTATION_RATE)
        fitness_histories_2.append(fitness_history)
        best_individual = min(population, key=lambda ind: calculate_fitness(ind, cities))
        plot_route(cities, best_individual, f"Best Route - {label}")

    generations = list(range(NUM_GENERATIONS))
    plot_fitness(generations, fitness_histories_2, labels_2)

    # Resultados del Experimento 2
    for label, fitness_history in zip(labels_2, fitness_histories_2):
        print(f"Final fitness for {label}: {fitness_history[-1]}")