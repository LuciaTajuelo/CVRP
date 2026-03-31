import numpy as np
from sklearn.cluster import KMeans
from src.models import VRPInstance, Solution
from src.algorithm.clarke_wright import clarke_wright

def clarke_wright_with_clustering(instance: VRPInstance, n_clusters: int = None) -> Solution:
    """
    Clarke & Wright con clusterización previa y manejo de clientes que quedan fuera por capacidad.
    """
    n_customers = len(instance.customers)
    
    # Determinar número de clusters
    if n_clusters is None:
        n_clusters = max(1, n_customers // instance.vehicle_capacity)
    
    # 1️⃣ Clusterizar clientes geográficamente
    coords = np.array([[c.x, c.y] for c in instance.customers])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(instance.customers[idx])
    
    # 2️⃣ Aplicar Clarke & Wright dentro de cada cluster
    solution = Solution(instance)
    unrouted_customers = []

    for cluster_id, cluster_customers in clusters.items():
        if not cluster_customers:
            continue
        
        # Crear sub-instancia usando el constructor correcto
        sub_instance = VRPInstance(
            name=f"{instance.name}_cluster{cluster_id}",
            depot=instance.depot,
            customers=cluster_customers,
            vehicle_capacity=instance.vehicle_capacity,
            best_known_cost=1,
            num_vehicles=instance.max_vehicles,
            problem_type = instance.problem_type
        )
        sub_solution = clarke_wright(sub_instance)
        
        # Añadir rutas al resultado final
        for route in sub_solution.routes:
            route_load = sum(instance.get_customer(c).demand for c in route)
            if route_load <= instance.vehicle_capacity:
                solution.add_route(route)
            else:
                # Si alguna ruta excede capacidad (caso raro), separar clientes
                unrouted_customers.extend(route)
    
    # 3️⃣ Intentar insertar clientes que quedaron fuera en rutas existentes
    still_unrouted = []
    for c_idx in unrouted_customers:
        customer = instance.get_customer(c_idx)
        inserted = False
        for route in solution.routes:
            route_load = sum(instance.get_customer(x).demand for x in route)
            if route_load + customer.demand <= instance.vehicle_capacity:
                route.append(c_idx)
                inserted = True
                break
        if not inserted:
            still_unrouted.append(c_idx)
    
    # 4️⃣ Crear nuevas rutas para los clientes que aún no se pudieron insertar
    for c_idx in still_unrouted:
        solution.add_route([c_idx])
    
    solution.compute_total_cost()
    solution.validate()
    return solution
