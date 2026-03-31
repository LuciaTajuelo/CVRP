import json
import os
from typing import List
from src.models import Customer, VRPInstance

def read_vrp_instance(input_path, bks_dir) -> VRPInstance:
    """
    Lee una instancia de VRP desde varios formatos y devuelve un objeto VRPInstance.
    Soporta:
      - Formato CMT (TXT)
      - JSON con clientes {"client":..., "lat":..., "lon":..., "demand":...}
    """
    _, ext = os.path.splitext(input_path)
    customers: List[Customer] = []
    depot = None
    capacity = None
    name = os.path.basename(input_path)

    # --- Formato CMT ---
    if ext.lower() in [".vrp"]:
        problem_type = 'benchmark'
        section = None
        coords = []
        demands = {}
        num_vehicles = None
        with open(input_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("COMMENT"):
                    continue

                if line.startswith("NAME"):
                    name = line.split(":")[1].strip()
                elif line.startswith("CAPACITY"):
                    capacity = int(line.split(":")[1].strip())
                elif line.startswith("NODE_COORD_SECTION"):
                    section = "COORDS"
                elif line.startswith("DEMAND_SECTION"):
                    section = "DEMANDS"
                elif line.startswith("DEPOT_SECTION"):
                    section = "DEPOT"
                elif line.startswith("EOF"):
                    break
                elif section == "COORDS":
                    parts = line.split()
                    if len(parts) == 3:
                        idx = int(parts[0])
                        x, y = float(parts[1]), float(parts[2])
                        coords.append((idx, x, y))
                elif section == "DEMANDS":
                    parts = line.split()
                    if len(parts) == 2:
                        idx, demand = int(parts[0]), int(parts[1])
                        demands[idx] = demand
                elif section == "DEPOT":
                    if line != "-1":
                        depot_id = int(line)
                        depot_data = next((c for c in coords if c[0] == depot_id), None)
                        if depot_data:
                            depot = (depot_data[1], depot_data[2])

        for idx, x, y in coords:
            if idx != depot_id:
                demand = demands.get(idx, 0)
                customers.append(Customer(idx-1, x, y, demand))

    # --- Formato JSON ---
    elif ext.lower() == ".json":
        problem_type = 'company'

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            client_list = data.get("clients", data)  # soporta lista directa
            # Agregar clientes, saltando el depósito
            for i, c in enumerate(client_list[1:], start=1):
                customers.append(Customer(i, float(c["lat"]), float(c["lon"]), int(c.get("demand", 0))))
            # Asumimos depósito en la primera posición
            depot = (float(client_list[0]["lat"]), float(client_list[0]["lon"]))
            capacity = data.get("meta", {}).get("vehicle_capacity", None)
            num_vehicles = data.get("meta", {}).get("num_vehicles", None)
        
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # --- Validación ---
    if not (capacity and depot and customers):
        raise ValueError(f"Error al leer VRPInstance: faltan datos esenciales en {input_path}")

    best_known_solution_distance = None
    if bks_dir:
        best_known_solution_distance = extract_cost_from_file(bks_dir)
    else:
        best_known_solution_distance = 1
    
    instance_name = os.path.splitext(name)[0]

    instance = VRPInstance(instance_name, depot, customers, capacity, best_known_solution_distance, num_vehicles, problem_type)
    return instance


def extract_cost_from_file(file_path: str) -> float:
    """
    Lee un archivo de solución CVRP y extrae el coste total.
    Se asume que la línea del coste empieza con "Cost".
    """
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Cost"):
                # La línea es algo como: "Cost 835.262"
                parts = line.split()
                try:
                    cost = float(parts[1])
                    return cost
                except (IndexError, ValueError):
                    raise ValueError(f"No se pudo interpretar el coste en la línea: {line}")
    raise ValueError("No se encontró la línea de coste en el archivo.")
