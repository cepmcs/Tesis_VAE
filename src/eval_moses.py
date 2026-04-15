"""
Evaluación de moléculas generadas usando MOSES.
Ejecutar en el entorno moses_fork:
    conda run -n moses_fork python eval_moses.py
"""
import argparse
import json
import os
import moses
import moses.metrics.metrics as metrics_module
import numpy as np
from scipy.spatial.distance import cosine

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parche temporal para corregir el overflow de float32 en moses/scipy
def safe_cos_distance(u, v):
    return cosine(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))

metrics_module.cos_distance = safe_cos_distance

def load_smiles(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Evaluar SMILES generados con MOSES")
    parser.add_argument("--gen", default=os.path.join(ROOT_DIR, "outputs", "generated_smiles.txt"),
                        help="Archivo TXT con 1 SMILES por línea")
    parser.add_argument("--out", default=os.path.join(ROOT_DIR, "outputs", "metrics.json"),
                        help="Archivo JSON de salida con métricas")
    args = parser.parse_args()

    # Cargar SMILES generados
    gen = load_smiles(args.gen)
    print(f"Moléculas cargadas: {len(gen)}")

    # Calcular métricas MOSES
    print("Calculando métricas (puede tardar unos minutos)...")
    metrics = moses.get_all_metrics(gen, n_jobs=1)

    # Mostrar resultados
    print("\n=== Métricas MOSES ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:>20s}: {value:.4f}")
        else:
            print(f"  {name:>20s}: {value}")

    # Guardar en JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"\nMétricas guardadas en: {args.out}")

if __name__ == "__main__":
    main()
