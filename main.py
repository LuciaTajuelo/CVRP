import os
import sys
import pandas as pd
from src.utils import *

def main(list_of_datasets = None):

    if not list_of_datasets:
        print("No datasets provided")
        return

    for dataset in list_of_datasets:

        DATA_FOLDER = f"data/benchmark/datasets/{dataset}"
        BKS_FOLDER = None
        RESULTS_DIR = f"data/benchmark/results/{dataset}"

        os.makedirs(RESULTS_DIR, exist_ok=True)

        results_df = pd.DataFrame()

        instance_files = sorted(
            f for f in os.listdir(DATA_FOLDER) if f.endswith(".vrp")
        )

        for vrp_file in instance_files:
            print("\n" + "=" * 80)
            print(f"Ejecutando instancia: {vrp_file}")

            data_path = os.path.join(DATA_FOLDER, vrp_file)

            try:
                solutions = run_experiment(
                    data_dir=data_path,
                    bks_dir=BKS_FOLDER,
                    results_dir=RESULTS_DIR
                )

                row_df = build_results_df(
                    solutions,
                    instance_name=vrp_file,
                    bks=solutions[list(solutions.keys())[0]].instance.best_known_cost
                )

                results_df = pd.concat([results_df, row_df], ignore_index=True)

                results_df.to_csv(
                    os.path.join(RESULTS_DIR, f"results_{dataset}.csv"),
                    index=False
                )

            except Exception as e:
                print(f"Error en {vrp_file}: {e}")

        print("\n=== Resumen final ===")
        print(results_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <dataset1> [dataset2 ...]")
        sys.exit(1)
    main(sys.argv[1:])