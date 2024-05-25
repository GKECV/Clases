import numpy as np
import matplotlib.pyplot as plt


def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights, returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility


def hill_climbing(returns, cov_matrix, iterations=1000, step_size=0.01):
    num_assets = len(returns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    best_return, best_volatility = portfolio_performance(weights, returns, cov_matrix)

    returns_list = [best_return]
    volatility_list = [best_volatility]

    for _ in range(iterations):
        new_weights = weights + np.random.uniform(-step_size, step_size, num_assets)
        new_weights = np.clip(new_weights, 0, 1)
        new_weights /= np.sum(new_weights)
        new_return, new_volatility = portfolio_performance(new_weights, returns, cov_matrix)

        if new_return / new_volatility > best_return / best_volatility:
            weights = new_weights
            best_return, best_volatility = new_return, new_volatility

        returns_list.append(best_return)
        volatility_list.append(best_volatility)

    return weights, returns_list, volatility_list


def simulated_annealing(returns, cov_matrix, iterations=1000, initial_temp=100, cooling_rate=0.99):
    num_assets = len(returns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    best_return, best_volatility = portfolio_performance(weights, returns, cov_matrix)
    current_temp = initial_temp

    returns_list = [best_return]
    volatility_list = [best_volatility]

    for _ in range(iterations):
        new_weights = weights + np.random.uniform(-0.1, 0.1, num_assets)
        new_weights = np.clip(new_weights, 0, 1)
        new_weights /= np.sum(new_weights)
        new_return, new_volatility = portfolio_performance(new_weights, returns, cov_matrix)

        if new_return / new_volatility > best_return / best_volatility:
            weights = new_weights
            best_return, best_volatility = new_return, new_volatility
        else:
            acceptance_probability = np.exp(
                (new_return / new_volatility - best_return / best_volatility) / current_temp)
            if acceptance_probability > np.random.rand():
                weights = new_weights
                best_return, best_volatility = new_return, new_volatility

        current_temp *= cooling_rate

        returns_list.append(best_return)
        volatility_list.append(best_volatility)

    return weights, returns_list, volatility_list


# Datos de ejemplo
returns = np.array([0.12, 0.18, 0.15])
cov_matrix = np.array([[0.005, -0.010, 0.004],
                       [-0.010, 0.040, -0.002],
                       [0.004, -0.002, 0.023]])

# Hill Climbing
hc_weights, hc_returns_list, hc_volatility_list = hill_climbing(returns, cov_matrix)

# Simulated Annealing
sa_weights, sa_returns_list, sa_volatility_list = simulated_annealing(returns, cov_matrix)

# Graficar resultados
plt.figure(figsize=(14, 10))

# Hill Climbing
plt.subplot(2, 2, 1)
plt.plot(hc_returns_list, label='Rendimiento')
plt.title('Rendimiento del Portafolio (Hill Climbing)')
plt.xlabel('Iteraciones')
plt.ylabel('Rendimiento')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(hc_volatility_list, label='Volatilidad', color='red')
plt.title('Volatilidad del Portafolio (Hill Climbing)')
plt.xlabel('Iteraciones')
plt.ylabel('Volatilidad')
plt.legend()

# Simulated Annealing
plt.subplot(2, 2, 3)
plt.plot(sa_returns_list, label='Rendimiento')
plt.title('Rendimiento del Portafolio (Simulated Annealing)')
plt.xlabel('Iteraciones')
plt.ylabel('Rendimiento')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(sa_volatility_list, label='Volatilidad', color='red')
plt.title('Volatilidad del Portafolio (Simulated Annealing)')
plt.xlabel('Iteraciones')
plt.ylabel('Volatilidad')
plt.legend()

plt.tight_layout()
plt.show()

print("Pesos óptimos (Hill Climbing):", hc_weights)
print("Rendimiento óptimo (Hill Climbing):", hc_returns_list[-1])
print("Volatilidad óptima (Hill Climbing):", hc_volatility_list[-1])

print("Pesos óptimos (Simulated Annealing):", sa_weights)
print("Rendimiento óptimo (Simulated Annealing):", sa_returns_list[-1])
print("Volatilidad óptima (Simulated Annealing):", sa_volatility_list[-1])