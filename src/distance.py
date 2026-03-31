import numpy as np
from typing import Dict, List
from .models import Customer, VRPInstance


def calculate_distance_matrix(customers: Dict[int, Customer]) -> np.ndarray:
    """Calcula la matriz de distancias entre todos los nodos.
    
    Args:
        customers: Diccionario {id: Customer} con todos los nodos
        
    Returns:
        np.ndarray: Matriz de distancias donde D[i,j] es distancia de i a j
    """
    n = len(customers)
    matrix = np.zeros((n, n))
    
    for i in customers:
        for j in customers:
            if i != j:
                matrix[i,j] = customers[i].distance_to(customers[j])
    
    return matrix


def get_route_length(route: List[int], instance: VRPInstance) -> float:
    """Calcula la longitud total de una ruta.
    
    Args:
        route: Lista de IDs de nodos en orden de visita
        instance: Instancia VRP con la matriz/info de distancias
        
    Returns:
        float: Suma de las distancias entre nodos consecutivos
    """
    if not route:
        return 0.0
        
    total = 0.0
    for i in range(len(route)-1):
        total += instance.get_distance(route[i], route[i+1])
    
    # Añadir retorno al depósito si la ruta no empieza/termina en él
    if route[0] != instance.depot_id:
        total += instance.get_distance(instance.depot_id, route[0])
    if route[-1] != instance.depot_id:
        total += instance.get_distance(route[-1], instance.depot_id)
        
    return total