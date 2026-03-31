from ortools.linear_solver import pywraplp
from src.models import Solution

def solve_cvrp_branch_and_cut(instance, time_limit_seconds=60*5):
    # --- Datos del problema ---
    locations = [instance.depot] + [c.coords() for c in instance.customers]
    demands = [0] + [c.demand for c in instance.customers]
    vehicle_capacity = instance.vehicle_capacity
    num_vehicles = int((len(locations) - 1) / 2)
    n_nodes = len(locations)

    # Distancia entera
    dist_matrix = [[int(round(x)) for x in row] for row in instance.dist_matrix]

    # --- Crear solver CBC (Branch-and-Cut) ---
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("No se pudo crear el solver CBC")

    # --- Límite de tiempo en milisegundos ---
    solver.SetTimeLimit(time_limit_seconds * 1000)

    # --- Variables binarias x[i,j] ---
    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                x[i,j] = solver.BoolVar(f'x_{i}_{j}')

    # --- Variables auxiliares para capacidad (MTZ) ---
    u = {}
    for i in range(1, n_nodes):
        u[i] = solver.NumVar(demands[i], vehicle_capacity, f'u_{i}')

    # --- Restricción: cada cliente visitado una vez ---
    for j in range(1, n_nodes):
        solver.Add(solver.Sum(x[i,j] for i in range(n_nodes) if i != j) == 1)
        solver.Add(solver.Sum(x[j,i] for i in range(n_nodes) if i != j) == 1)

    # --- Restricciones MTZ para subtours y capacidad ---
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i != j:
                solver.Add(u[i] + demands[j] - u[j] <= vehicle_capacity*(1 - x[i,j]))

    # --- Función objetivo: minimizar distancia total ---
    objective_terms = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                objective_terms.append(dist_matrix[i][j] * x[i,j])
    solver.Minimize(solver.Sum(objective_terms))

    # --- Resolver ---
    status = solver.Solve()

    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        if status == pywraplp.Solver.FEASIBLE:
            print(f"Límite de tiempo alcanzado. Mejor solución encontrada con coste {solver.Objective().Value()}")
        else:
            print("Coste óptimo:", solver.Objective().Value())

        # --- Reconstruir rutas ---
        solution_output = Solution(instance)
        visited = set()
        for v in range(num_vehicles):
            route = []
            current = 0  # inicio en depósito
            while True:
                next_node = None
                for j in range(1, n_nodes):  # solo clientes, excluye depósito
                    if (current,j) in x and x[current,j].solution_value() > 0.5 and j not in visited:
                        next_node = j
                        break
                if next_node is None:
                    break
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            if route:
                solution_output.add_route(route)
        solution_output.validate()

        return solution_output
    else:
        print("No se encontró solución factible.")