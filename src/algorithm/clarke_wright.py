# clarke_wright.py
from typing import List, Tuple
from src.models import VRPInstance, Solution

def clarke_wright(instance: VRPInstance) -> Solution:
    """
    Implementación básica del algoritmo Clarke & Wright (Savings) para CVRP.
    
    Parámetros:
    -----------
    instance : VRPInstance
        Instancia del problema CVRP.

    Retorna:
    --------
    solution : Solution
        Solución generada con Clarke & Wright.
    """

    n_customers = len(instance.customers)
    dist = instance.dist_matrix

    # 1️ Inicializar rutas individuales para cada cliente
    routes = {c.idx: [c.idx] for c in instance.customers}  # cliente -> ruta
    route_loads = {c.idx: c.demand for c in instance.customers}  # carga de cada ruta

    # 2️ Calcular los "savings" para cada par de clientes
    savings_list: List[Tuple[int, int, float]] = []
    for i in range(n_customers):
        for j in range(i+1, n_customers):
            ci = instance.customers[i]
            cj = instance.customers[j]
            saving = dist[0, i+1] + dist[0, j+1] - dist[i+1, j+1]
            savings_list.append((ci.idx, cj.idx, saving))

    # 3️ Ordenar los savings de mayor a menor
    savings_list.sort(key=lambda x: x[2], reverse=True)

    # 4️ Merge de rutas basado en savings
    for i, j, s in savings_list:
        ri = routes.get(i)
        rj = routes.get(j)

        # Si ambos clientes están en rutas diferentes
        if ri != rj:
            # Verificar que i esté al final de su ruta y j al inicio (o viceversa)
            if (ri[-1] == i and rj[0] == j) or (ri[0] == i and rj[-1] == j):
                # Verificar capacidad
                if route_loads[i] + route_loads[j] <= instance.vehicle_capacity:
                    # Hacer merge
                    if ri[-1] == i and rj[0] == j:
                        merged = ri + rj
                    elif ri[0] == i and rj[-1] == j:
                        merged = rj + ri
                    elif ri[0] == i and rj[0] == j:
                        merged = rj[::-1] + ri
                    else:
                        merged = ri + rj[::-1]

                    # Actualizar rutas y cargas
                    for cid in merged:
                        routes[cid] = merged
                        route_loads[cid] = sum(instance.get_customer(x).demand for x in merged)

    # 5️ Construir solución final eliminando duplicados
    unique_routes = []
    seen = set()
    for r in routes.values():
        tup = tuple(r)
        if tup not in seen:
            unique_routes.append(r)
            seen.add(tup)

    solution = Solution(instance)
    for r in unique_routes:
        solution.add_route(r)
    solution.compute_total_cost()

    # Ejecutar validaciones y calcular coste total
    solution.validate()

    return solution
