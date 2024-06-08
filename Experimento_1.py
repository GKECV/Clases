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
    return sum(distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1)) + distance(
        cities[route[-1]], cities[route[0]])


# Métodos de Selección
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    selection_probs = [f / total_fitness for f in fitness]
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs)
    return [population[i] for i in selected_indices]


def rank_based_selection(population, fitness):
    sorted_indices = np.argsort(fitness)
    ranks = range(1, len(fitness) + 1)
    selection_probs = [rank / sum(ranks) for rank in ranks]
    selected_indices = np.random.choice(sorted_indices, size=len(population), p=selection_probs)
    return [population[i] for i in selected_indices]


def fitness_scaling_selection(population, fitness):
    scaled_fitness = [f ** 2 for f in fitness]
    total_scaled_fitness = sum(scaled_fitness)
    selection_probs = [f / total_scaled_fitness for f in scaled_fitness]
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs)
    return [population[i] for i in selected_indices]


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


# Función de inicialización aleatoria
def random_initialization(num_individuals, num_cities):
    population = []
    for _ in range(num_individuals):
        individual = np.random.permutation(num_cities).tolist()
        population.append(individual)
    return population


# Ejecución del algoritmo genético
def run_ga(population, cities, selection_method, fitness_func, num_generations, mutation_rate):
    fitness_history = []
    for generation in range(num_generations):
        # Evaluar el fitness de la población
        fitness_values = [fitness_func(individual, cities) for individual in population]
        # Guardar el mejor fitness en cada generación
        fitness_history.append(min(fitness_values))

        # Seleccionar el mejor individuo
        best_individual = min(population, key=lambda ind: fitness_func(ind, cities))

        # Crear una nueva población y mantener el mejor individuo
        next_population = [best_individual]

        # Rellenar la nueva población con hijos
        while len(next_population) < len(population):
            # Selección de padres
            father = selection_method(population, fitness_values)
            mother = selection_method(population, fitness_values)

            # Cruce
            child = crossover(father[0], mother[0])

            # Mutación
            if np.random.rand() < mutation_rate:
                mutate(child, mutation_rate)

            # Añadir hijo a la nueva población
            next_population.append(child)

        # Reemplazar la población con la nueva
        population = next_population

    # Encontrar el mejor individuo de la última población
    best_individual = min(population, key=lambda ind: fitness_func(ind, cities))
    return best_individual, fitness_history


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
    plt.title("Comparación de Métodos de Selección")
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

    # Ejecución del Experimento 1
    fitness_histories_1 = []
    labels_1 = ["Roulette", "Rank-based", "Fitness Scaling", "Tournament"]
    selection_methods = [roulette_wheel_selection, rank_based_selection, fitness_scaling_selection,
                         tournament_selection]

    for selection_method, label in zip(selection_methods, labels_1):
        # Inicialización de población
        population = random_initialization(NUM_INDIVIDUALS, NUM_CITIES)
        # Ejecutar algoritmo genético
        best_individual, fitness_history = run_ga(population, cities, selection_method, calculate_fitness,
                                                  NUM_GENERATIONS, MUTATION_RATE)
        fitness_histories_1.append(fitness_history)
        # Mostrar la mejor ruta obtenida
        plot_route(cities, best_individual, f"Best Route - {label}")

    generations = list(range(NUM_GENERATIONS))
    # Mostrar la evolución del fitness
    plot_fitness(generations, fitness_histories_1, labels_1)

    # Resultados del Experimento 1
    for label, fitness_history in zip(labels_1, fitness_histories_1):
        print(f"Final fitness for {label}: {fitness_history[-1]}")
