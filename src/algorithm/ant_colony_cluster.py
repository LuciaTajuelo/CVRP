import numpy as np
import random
from sklearn.cluster import KMeans
from src.models import Solution, VRPInstance

def cluster_customers(instance: VRPInstance, n_clusters: int):
    """Agrupa clientes usando K-means según su posición."""
    coords = np.array([[c.x, c.y] for c in instance.customers])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(instance.customers[idx].idx)
    return clusters

def aco_vrp_clustered(instance: VRPInstance,  n_ants=50, n_iter=150, alpha=1, beta=5, rho=0.2):
    dist = instance.dist_matrix
    pheromone = np.ones_like(dist)
    best_solution = None

    n_clusters = max(1, len(instance.customers) // instance.vehicle_capacity)
    clusters = cluster_customers(instance, n_clusters)

    for iteration in range(n_iter):
        ant_solutions = []

        for k in range(n_ants):
            unvisited = set(c.idx for c in instance.customers)
            solution = Solution(instance)

            cluster_order = list(clusters.keys())
            random.shuffle(cluster_order)

            for cluster_id in cluster_order:
                cluster_customers_set = set(clusters[cluster_id]) & unvisited
                route = []
                load = 0
                current_node = 0  # depósito

                while cluster_customers_set:
                    feasible = [c for c in cluster_customers_set
                                if c != current_node and load + instance.get_customer(c).demand <= instance.vehicle_capacity]
                    if not feasible:
                        if route:
                            solution.add_route(route)
                        route = []
                        load = 0
                        current_node = 0
                        continue

                    probs = []
                    for c in feasible:
                        tau = pheromone[current_node, c] ** alpha
                        d = dist[current_node, c]
                        eta = (1 / d if d > 1e-6 else 1e6) ** beta
                        probs.append(tau * eta)

                    probs = np.array(probs)
                    total = probs.sum()
                    if total == 0 or not np.isfinite(total):
                        probs = np.ones_like(probs) / len(probs)
                    else:
                        probs = probs / total

                    next_c = random.choices(feasible, weights=probs)[0]
                    route.append(next_c)
                    load += instance.get_customer(next_c).demand
                    cluster_customers_set.remove(next_c)
                    unvisited.remove(next_c)
                    current_node = next_c

                if route:
                    solution.add_route(route)

            # Clientes restantes fuera de clusters por capacidad
            if unvisited:
                route = []
                load = 0
                current_node = 0
                while unvisited:
                    feasible = [c for c in unvisited if c != current_node and load + instance.get_customer(c).demand <= instance.vehicle_capacity]
                    if not feasible:
                        if route:
                            solution.add_route(route)
                        route = []
                        load = 0
                        current_node = 0
                        continue

                    # Selección simple para los restantes
                    next_c = feasible[0]
                    route.append(next_c)
                    load += instance.get_customer(next_c).demand
                    unvisited.remove(next_c)
                    current_node = next_c

                if route:
                    solution.add_route(route)

            solution.compute_total_cost()
            ant_solutions.append(solution)

        iteration_best = min(ant_solutions, key=lambda s: s.total_cost)
        if best_solution is None or iteration_best.total_cost < best_solution.total_cost:
            best_solution = iteration_best

        # Evaporación
        pheromone *= (1 - rho)

        # Refuerzo feromonas
        for route in iteration_best.routes:
            for i in range(len(route)):
                from_node = 0 if i == 0 else route[i - 1]
                to_node = route[i]
                pheromone[from_node, to_node] += 1 / iteration_best.total_cost
                pheromone[to_node, from_node] = pheromone[from_node, to_node]

    best_solution.validate()
    return best_solution
