# VRP Project

Este proyecto implementa algoritmos para resolver el Problema de Ruteo de Vehículos (VRP - Vehicle Routing Problem) utilizando diversas heurísticas y métodos exactos.

## Estructura del Proyecto

- `data/`: Contiene datasets de benchmark y resultados.
  - `benchmark/datasets/`: Instancias VRP en formato .vrp y soluciones óptimas en .sol.
  - `company/`: Datos personalizados de clientes y coordenadas.
- `models/`: Modelos entrenados (e.g., attention_actor.pt, attention_critic.pt).
- `src/`: Código fuente.
  - `algorithm/`: Implementaciones de algoritmos (Ant Colony, Clarke-Wright, GNN, etc.).
  - `models.py`: Definiciones de instancias VRP y soluciones.
  - `parser.py`: Parser para archivos .vrp.
  - `utils.py`: Utilidades para ejecutar experimentos.
  - `visualize.py`: Herramientas de visualización.
- `main.py`: Script principal para ejecutar experimentos.

## Requisitos

- Python 3.8+
- Pandas
- NumPy
- PyTorch (para algoritmos GNN)
- Otros: matplotlib (para visualización), ortools (para métodos exactos)

Instala las dependencias con:

```bash
pip install pandas numpy torch matplotlib ortools
```

## Instalación

1. Clona o descarga el repositorio.
2. Instala las dependencias: `pip install -r requirements.txt` (si existe, o manualmente).
3. Asegúrate de que los modelos en `models/` estén presentes.

## Uso

Ejecuta experimentos desde terminal pasando los datasets como argumentos:

```bash
python main.py dataset1 dataset2 dataset3
```

Esto ejecutará los algoritmos en los datasets A, B y M, guardando resultados en `data/benchmark/results/`.

### Datasets Disponibles

- A, B, CMT, F, M, P, X, X_300, X_900 (ver `data/benchmark/datasets/`).

### Algoritmos Implementados

- Clarke-Wright (Savings Algorithm)
- Ant Colony Optimization
- GNN (Graph Neural Network) con beam search
- Método exacto con branch and cut (usando OR-Tools)

## Resultados

Los resultados se guardan en CSV en `data/benchmark/results/<dataset>/results_<dataset>.csv`, incluyendo costos, tiempos y gaps.

## Troubleshooting

- **Error de import**: Asegúrate de que `src/` esté en el PYTHONPATH o ejecuta desde el directorio raíz.
- **Modelos faltantes**: Verifica que los archivos .pt en `models/` estén presentes.
- **Dependencias**: Instala todas las librerías requeridas.

## Contribución

Siéntete libre de contribuir mejoras o nuevos algoritmos.</content>
