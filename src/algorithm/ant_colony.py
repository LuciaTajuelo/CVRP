import numpy as np
import random
from src.models import Solution, VRPInstance

def aco_vrp(instance: VRPInstance, n_ants=50, n_iter=150, alpha=1, beta=5, rho=0.1):
    dist = instance.dist_matrix
    pheromone = np.ones_like(dist)  # matriz de feromonas
    best_solution = None
    
    n_nodes = dist.shape[0]

    for iteration in range(n_iter):
        ant_solutions = []

        for k in range(n_ants):
            unvisited = set(c.idx for c in instance.customers)
            solution = Solution(instance)
            route = []
            load = 0
            current_node = 0  # depósito

            while unvisited:
                # clientes factibles según capacidad y que no sean el nodo actual
                feasible = [c for c in unvisited if c != current_node and load + instance.get_customer(c).demand <= instance.vehicle_capacity]
                
                if not feasible:
                    if route:
                        solution.add_route(route)
                    route = []
                    load = 0
                    current_node = 0
                    continue

                # calcular probabilidades
                probs = []
                for c in feasible:
                    tau = pheromone[current_node, c] ** alpha
                    d = dist[current_node, c]
                    # proteger contra distancias cero o muy pequeñas
                    eta = (1 / d if d > 1e-6 else 1e6) ** beta
                    probs.append(tau * eta)

                probs = np.array(probs)
                total = probs.sum()
                if total == 0 or not np.isfinite(total):
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / total

                # seleccionar cliente siguiente
                next_c = random.choices(feasible, weights=probs)[0]

                route.append(next_c)
                load += instance.get_customer(next_c).demand
                unvisited.remove(next_c)
                current_node = next_c

            if route:
                solution.add_route(route)

            solution.compute_total_cost()
            ant_solutions.append(solution)

        # actualizar mejor global
        iteration_best = min(ant_solutions, key=lambda s: s.total_cost)
        if best_solution is None or iteration_best.total_cost < best_solution.total_cost:
            best_solution = iteration_best

        # evaporación
        pheromone *= (1 - rho)

        # refuerzo feromonas
        for route in iteration_best.routes:
            for i in range(len(route)):
                from_node = 0 if i == 0 else route[i-1]
                to_node = route[i]
                pheromone[from_node, to_node] += 1 / iteration_best.total_cost
                pheromone[to_node, from_node] = pheromone[from_node, to_node]

    best_solution.validate()
    return best_solution