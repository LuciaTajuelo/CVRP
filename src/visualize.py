import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path

from .models import VRPInstance, Solution


def plot_solution(
    instance: VRPInstance,
    solution: Solution,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Path] = None
) -> None:
    """Visualiza una solución VRP.
    
    Args:
        instance: Instancia del problema
        solution: Solución a visualizar
        title: Título opcional para el gráfico
        show: Si True, muestra el plot
        save_path: Ruta opcional donde guardar la imagen
    """
    plt.figure(figsize=(12, 8))
    
    # Plotear clientes
    xs = [c.x for c in instance.customers.values() if not c.is_depot]
    ys = [c.y for c in instance.customers.values() if not c.is_depot]
    plt.scatter(xs, ys, c='blue', s=50, label='Clientes')
    
    # Plotear depósito
    depot = instance.depot
    plt.scatter([depot.x], [depot.y], c='red', s=100, marker='s', label='Depósito')
    
    # Plotear rutas con diferentes colores
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
    for route, color in zip(solution.routes, colors):
        coords = [(instance.customers[i].x, instance.customers[i].y) for i in route]
        xs, ys = zip(*coords)
        plt.plot(xs, ys, c=color, linewidth=1, alpha=0.7)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Solución VRP - {instance.name}\nCoste total: {solution.cost:.2f}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()