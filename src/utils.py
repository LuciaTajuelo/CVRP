import os
import time
import pandas as pd
from src.parser import read_vrp_instance

from src.models import VRPInstance
from src.algorithm.clarke_wright import clarke_wright
from src.algorithm.clarke_wright_cluster import clarke_wright_with_clustering
from src.algorithm.gnn import gnn_beam_guided
from src.algorithm.exact_model_or_tools import solve_cvrp_branch_and_cut
from src.algorithm.ant_colony import aco_vrp
from src.algorithm.ant_colony_cluster import aco_vrp_clustered
from src.models import VRPInstance, Solution

def run_model(model_name: str, model_func, instance: VRPInstance, results_dir: str, initial_solution=None):
    """
    Función genérica para ejecutar cualquier modelo/heurística VRP.
    - model_name: nombre que se usará para carpeta y archivo.
    - model_func: función del modelo (clarke_wright, gnn, solve_cvrp, etc.).
    - instance: instancia VRP.
    - results_dir: directorio donde guardar resultados.
    - initial_solution: solución inicial opcional.
    """
    start = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # Llamada genérica
    if initial_solution is None:
        solution = model_func(instance)
    else:
        solution = model_func(initial_solution)

    if solution is None:
        solution = Solution(instance)
        solution.execution_time = time.time() - start
        print(f" [{model_name}]  No encuentra solución")

    else:
        solution.execution_time = time.time() - start
        print(f"FO = {solution.total_cost:.2f} | N rutas = {len(solution.routes)} | [{model_name}]  | Tiempo {solution.execution_time}")

    # Guardado
    solution.save_json(results_dir, model_name)
    
    return solution

def run_experiment(data_dir: str, bks_dir: str, results_dir: str):

    instance = read_vrp_instance(data_dir, bks_dir)
    print(f"\nInstancia: {instance.name} | Clientes: {len(instance.customers)} | Capacidad: {instance.vehicle_capacity} | Vehiculos: {instance.max_vehicles} | BKS: {instance.best_known_cost}")

    solutions = {}

    solutions["aco_vrp"] = run_model("aco_vrp", aco_vrp, instance, results_dir)
    solutions["aco_vrp_clustered"] = run_model("aco_vrp_clustered", aco_vrp_clustered, instance, results_dir)
    
    solutions["clarke_wright"] = run_model("clarke_wright", clarke_wright, instance, results_dir)
    # solutions["clarke_wright_with_clustering"] = run_model("clarke_wright_with_clustering", clarke_wright_with_clustering, instance, results_dir)
    solutions["solve_cvrp_branch_and_cut"] = run_model("or_tools", solve_cvrp_branch_and_cut, instance, results_dir)

    solutions["gnn_beam_guided"] = run_model("gnn_beam_guided", gnn_beam_guided, instance, results_dir)
    
    return solutions

def build_results_df(solutions: dict, instance_name: str, bks: float) -> pd.DataFrame:
    """
    Construye un DataFrame de una fila con los resultados de todas las heurísticas.
    """
    row = {'instancia': instance_name, 'bks': bks}
    metrics = ['total_cost', 'gap', 'execution_time', 'num_routes']

    for key, sol in solutions.items():
        for metric in metrics:
            if metric == 'num_routes':
                value = sol.num_routes()
            else:
                value = getattr(sol, metric)
            row[f"{key}_{metric}"] = value

    return pd.DataFrame([row])