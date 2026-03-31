import json
import random
import os



def generate_and_save_demands(
    clients,
    output_json_path,
    num_vehicles=5,
    vehicle_capacity=800,
    min_demand=10,
    max_demand=300,
    seed=None
):
    """
    Genera demandas aleatorias para los clientes y guarda el dataset en JSON.
    El depósito (demand=0) no recibe demanda.
    Garantiza que la suma de demandas no supere la capacidad total de los vehículos.
    """

    if seed is not None:
        random.seed(seed)

    # Excluir depósitos de la generación de demanda
    clients_to_assign = [c for c in clients if c.get("demand", None) != 0]
    n = len(clients_to_assign)
    total_capacity = num_vehicles * vehicle_capacity

    if n * min_demand > total_capacity:
        raise ValueError(
            f"Capacidad total insuficiente ({total_capacity}) para asignar el mínimo ({min_demand}) a {n} clientes."
        )

    # Generar demandas proporcionales
    remaining_capacity = total_capacity - n * min_demand
    random_weights = [random.random() for _ in range(n)]
    sum_weights = sum(random_weights)

    demands = [min_demand + int(w / sum_weights * remaining_capacity) for w in random_weights]

    # Ajuste final para que la suma no exceda la capacidad total
    diff = total_capacity - sum(demands)
    for j in range(diff):
        demands[j % n] += 1

    # Asignar demandas a los clientes
    j = 0
    for client in clients:
        if client.get("demand", None) == 0:
            continue  # Saltar depósito
        client["demand"] = demands[j]
        j += 1

    data = {
        "clients": clients,
        "meta": {
            "num_vehicles": num_vehicles,
            "vehicle_capacity": vehicle_capacity,
            "total_capacity": total_capacity,
            "total_demand": sum(demands),
            "min_demand": min_demand,
            "max_demand": max_demand,
            "seed": seed
        }
    }

    # Crear carpeta de salida si no existe
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Guardar archivo
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Archivo guardado en: {output_json_path}")
# ------------------------------
# Ejemplo: generar varios datasets sin sobrescribir
# ------------------------------

# Leer clientes desde archivo base
with open("data/company/clients.json", "r", encoding="utf-8") as f:
    base_clients = json.load(f)["clients"]

datasets = [
    {"name": "low", "min_demand": 0, "max_demand": 50},
    {"name": "medium", "min_demand": 10, "max_demand": 100},
    {"name": "high", "min_demand": 50, "max_demand": 700},
]

num_variations = 3  # cuántos archivos por rango

depot = {"client": "Depot", "lat": 41.469592, "lon": 2.029750, "demand": 0}

for scenario in datasets:
    for i in range(num_variations):
        # Clonar clientes para no modificar el original
        clients_copy = [dict(c) for c in base_clients]
        
        # Añadir el depósito al inicio
        clients_copy.insert(0, depot)
        
        output_file = f"data/company/datasets/clients_{scenario['name']}_{i+1}.json"
        seed = i + 100
        generate_and_save_demands(
            clients=clients_copy,
            output_json_path=output_file,
            num_vehicles=5,
            vehicle_capacity=800,
            min_demand=scenario["min_demand"],
            max_demand=scenario["max_demand"],
            seed=seed
        )