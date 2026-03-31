"""
models.py — Estructuras de datos base para el problema CVRP
Autor: Lu
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from math import sqrt
import pickle
import os, json
import matplotlib.pyplot as plt
import random
from geopy.distance import geodesic
import folium

# ==========================================================
# Clase Customer
# ==========================================================
class Customer:
    """Representa un cliente o nodo en la instancia VRP."""

    def __init__(self, idx: int, x: float, y: float, demand: int):
        self.idx = idx          # Identificador del cliente
        self.x = x              # Coordenada X
        self.y = y              # Coordenada Y
        self.demand = demand    # Demanda (cantidad a entregar)

    def coords(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self):
        return f"Customer(id={self.idx}, demand={self.demand}, loc=({self.x},{self.y}))"


# ==========================================================
# Clase Vehicle
# ==========================================================
class Vehicle:
    """Representa un vehículo con su capacidad y ruta asignada."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.load = 0
        self.route: List[int] = []  # Lista de IDs de clientes visitados (orden)

    def add_customer(self, customer: Customer):
        """Agrega un cliente a la ruta si hay capacidad disponible."""
        if self.load + customer.demand <= self.capacity:
            self.route.append(customer.idx)
            self.load += customer.demand
        else:
            raise ValueError("Capacidad del vehículo excedida.")

    def reset(self):
        self.route.clear()
        self.load = 0

    def __repr__(self):
        return f"Vehicle(cap={self.capacity}, load={self.load}, route={self.route})"


# ==========================================================
# Clase VRPInstance
# ==========================================================

class VRPInstance:
    """Contiene toda la información de una instancia CVRP."""

    def __init__(self, name: str, depot: Tuple[float, float], customers: List[Customer], vehicle_capacity: int, best_known_cost: float, num_vehicles: int, problem_type: str):
        self.name = name
        self.depot = depot                    # Coordenadas del depósito
        self.customers = customers            # Lista de clientes
        self.vehicle_capacity = vehicle_capacity
        self.best_known_cost = best_known_cost
        self.problem_type = problem_type
        self.max_vehicles = num_vehicles
        self.dist_matrix = self._compute_distance_matrix(problem_type)

    def num_customers(self) -> int:
        return len(self.customers)

    def get_customer(self, idx: int) -> Customer:
        return next(c for c in self.customers if c.idx == idx)

    def _compute_distance_matrix(self, problem_type) -> np.ndarray:
        """
        Calcula la matriz de distancias entre todos los clientes y el depósito.
        Nodo 0 = depósito.
        - VRP → Euclidiana
        - JSON → Geodésica en km
        """
        n_customers = len(self.customers)
        n_nodes = n_customers + 1
        dist_matrix = np.zeros((n_nodes, n_nodes))
        coords = [self.depot] + [c.coords() for c in self.customers]

        # Seleccionamos la función de distancia una sola vez
        if problem_type == 'company':
            distance_func = lambda a, b: geodesic(a, b).kilometers
        else:
            distance_func = lambda a, b: sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        for i in range(n_nodes):
            for j in range(i, n_nodes):
                dist = distance_func(coords[i], coords[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # simetría

        return dist_matrix

    def __repr__(self):
        return f"VRPInstance(name={self.name}, customers={len(self.customers)}, capacity={self.vehicle_capacity})"
    
# ==========================================================
# Clase Solution
# ==========================================================
class Solution:
    """Representa una solución al problema CVRP."""

    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.best_known_cost = round(instance.best_known_cost, 3)
        self.gap = None
        self.routes: List[List[int]] = []   # Lista de rutas (cada una lista de IDs de clientes)
        self.total_cost: Optional[float] = None
        self.execution_time: Optional[float] = None

        # Estado de validación
        self.is_feasible_flag: Optional[bool] = None
        self.all_customers_served_flag: Optional[bool] = None
        self.capacity_feasible_flag: Optional[bool] = None


    def add_route(self, route: List[int]):
        """Agrega una ruta (se asume válida)."""
        self.routes.append(route)
        self.validate()

    def num_routes(self) -> int:
        return len(self.routes)

    def all_customers(self) -> List[int]:
        """Devuelve la lista de todos los clientes servidos en la solución."""
        return [cid for route in self.routes for cid in route]

    def __repr__(self):
        cost = f"{self.total_cost:.2f}" if self.total_cost else "?"
        return f"Solution(routes={len(self.routes)}, cost={cost})"

    def to_dict(self) -> Dict:
        return {
            "instance": self.instance.name,
            "routes": self.routes,
            "total_cost": self.total_cost,
            "is_feasible": self.is_feasible_flag,
            "all_customers_served": self.all_customers_served_flag
        }
    
    def save_cvrp_map(self, results_dir, model_name):
        """
        Guarda un mapa de la solución CVRP como PNG.
        Funciona con:
        - self.routes: rutas con IDs de clientes
        - self.instance.customers: lista de Customer
        - self.instance.depot: tuple (x, y)
        """
        depot_x, depot_y = self.instance.depot
        customers = self.instance.customers

        # Mapear ID de cliente -> objeto Customer
        id_to_customer = {c.idx: c for c in customers}

        plt.figure(figsize=(8, 6))

        # Pintar depósito
        plt.scatter(depot_x, depot_y, c='red', marker='s', s=100, label='Depot')

        # Colores aleatorios para cada ruta
        colors = [(random.random(), random.random(), random.random()) for _ in self.routes]

        for route, color in zip(self.routes, colors):
            x = [depot_x] + [id_to_customer[cid].x for cid in route] + [depot_x]
            y = [depot_y] + [id_to_customer[cid].y for cid in route] + [depot_y]

            plt.plot(x, y, color=color, linewidth=2, alpha=0.7)
            plt.scatter(x[1:-1], y[1:-1], color=color, s=50)  # clientes

        plt.title("CVRP Solution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)

        # Guardar imagen
        path = os.path.join(results_dir, model_name, f"{self.instance.name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"Mapa guardado en {path}")

    def save_folium_map(self, results_dir, model_name):
        """
        Genera un mapa interactivo con Folium mostrando las rutas de la solución.

        """
        depot_lat, depot_lon = self.instance.depot
        m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

        # Colores para las rutas
        colors = ["red", "blue", "green", "purple", "orange", "darkred", 
                "lightred", "beige", "darkblue", "darkgreen", "cadetblue", 
                "darkpurple", "white", "pink", "lightblue", "lightgreen"]

        id_to_customer = {c.idx: c for c in self.instance.customers}

        for ridx, route in enumerate(self.routes):
            # Coordenadas de la ruta, incluyendo inicio y fin en depósito
            route_coords = [(depot_lat, depot_lon)]
            for cid in route:
                c = id_to_customer[cid]
                route_coords.append((c.x, c.y))
            route_coords.append((depot_lat, depot_lon))

            # Dibujar línea de la ruta
            folium.PolyLine(route_coords, color=colors[ridx % len(colors)], weight=3, opacity=0.7).add_to(m)

            # Dibujar clientes
            for lat, lon in route_coords[1:-1]:
                folium.CircleMarker(location=(lat, lon), radius=4, color='black', fill=True).add_to(m)

        # Crear carpeta si no existe
        map_path = os.path.join(results_dir, model_name, f"{self.instance.name}_map.html")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)

        # Guardar mapa
        m.save(map_path)
        print(f"Mapa interactivo guardado en {map_path}")
        
    def save_pickle(self, results_dir, model_name, instance_name):

        path = os.path.join(results_dir, model_name, f"{instance_name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def save_json(self, results_dir, model_name):

        path = os.path.join(results_dir, model_name, f"{self.instance.name}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.instance.problem_type == 'benchmark':
            self.save_cvrp_map(results_dir, model_name)
        elif self.instance.problem_type == 'company':
            self.save_folium_map(results_dir, model_name)
        else:
            print( 'error building solution map')
    # ======================================================
    # Validaciones
    # ======================================================

    def all_customers_served(self) -> bool:
        """Verifica que todos los clientes de la instancia se visiten exactamente una vez."""
        served = self.all_customers()
        expected = {c.idx for c in self.instance.customers}
        return set(served) == expected and len(served) == len(self.instance.customers)
    
    def check_route_capacity(self) -> bool:
        """Verifica que la ruta no exceda la capacidad del vehículo."""
        for r in self.routes:
            load = sum(self.instance.get_customer(cid).demand for cid in r)
            if load > self.instance.vehicle_capacity:
                return False
        return load <= self.instance.vehicle_capacity
    # ======================================================
    # 🔹 Validación completa
    # ======================================================
    def validate(self):
        """
        Ejecuta todas las validaciones y calcula el coste total,
        actualizando los flags de estado.
        """
        self.all_customers_served_flag = self.all_customers_served()
        self.capacity_feasible_flag =  self.check_route_capacity()
        self.is_feasible_flag = self.all_customers_served_flag and self.capacity_feasible_flag
        self.compute_total_cost()
        # falta comprobar el número de vehículos
        return self.is_feasible_flag

    # ======================================================
    # Cálculo del coste total
    # ======================================================
    def compute_total_cost(self) -> float:
        """Calcula el coste total de la solución sumando la distancia de cada ruta, incluyendo la ida y vuelta al depósito."""
        total = 0.0
        dist = self.instance.dist_matrix

        for route in self.routes:
            if not route:
                continue
            prev = 0  # depósito al inicio
            for cid in route:
                # idx en la matriz: 0 = depósito, clientes en posiciones 1..n
                pos = next(i+1 for i, c in enumerate(self.instance.customers) if c.idx == cid)
                total += dist[prev, pos]
                prev = pos
            total += dist[prev, 0]  # volver al depósito

        self.total_cost = total
        self.gap = (total - self.best_known_cost) / self.best_known_cost * 100
        return round(total, 3)
    
    def init_dummy_solution(self):
        """
        Crea una solución trivial: una ruta por cliente,
        cada una visitando únicamente a ese cliente.
        """
        self.routes = [[c.idx] for c in self.instance.customers]
        self.validate()