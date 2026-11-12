"""
Evaluación de moléculas generadas usando MOSES.
Ejecutar en el entorno moses_fork:
    conda run -n moses_fork python eval_moses.py

Itera sobre todos los *_generated.txt en outputs/
- Guarda un JSON individual por modelo en outputs/eval_moses/
- Genera un resumen .txt con las métricas clave
"""
import json
import os
import glob
import moses
import moses.metrics.metrics as metrics_module
import numpy as np
from scipy.spatial.distance import cosine

# Directorio raíz del proyecto (un nivel arriba de scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
EVAL_DIR = os.path.join(OUTPUTS_DIR, "eval_moses")

# Parche temporal para corregir el overflow de float32 en moses/scipy
def safe_cos_distance(u, v):
    return cosine(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))

metrics_module.cos_distance = safe_cos_distance


def load_smiles(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_single(gen_path, json_path):
    """Evalúa un archivo de SMILES y guarda métricas en JSON. Retorna dict."""
    gen = load_smiles(gen_path)
    print(f"  Moléculas cargadas: {len(gen)}", flush=True)

    print("  Calculando métricas...", flush=True)
    metrics = moses.get_all_metrics(gen, n_jobs=1)

    # Mostrar resultados
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"    {name:>20s}: {value:.4f}", flush=True)
        else:
            print(f"    {name:>20s}: {value}", flush=True)

    # Guardar JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)

    print(f"  JSON guardado: {json_path}", flush=True)
    return metrics


def format_value(val):
    """Formatea un valor para el resumen."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Buscar todos los _generated.txt
    gen_files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*_generated.txt")))

    if not gen_files:
        print(f"No se encontraron archivos *_generated.txt en {OUTPUTS_DIR}")
        return

    print(f"=== Evaluación MOSES ===", flush=True)
    print(f"Archivos: {len(gen_files)}", flush=True)
    print("", flush=True)

    # Métricas clave para el resumen
    # Claves exactas de MOSES (verificadas con metrics.json)
    summary_keys = [
        ("valid", "Validity"),
        ("unique@10000", "Uniqueness"),
        ("Novelty", "Novelty"),
        ("Filters", "Filters"),
        ("FCD/Test", "FCD"),
        ("IntDiv", "IntDiv"),
        ("SA", "SA"),
        ("QED", "QED"),
    ]

    all_results = []

    for i, gen_path in enumerate(gen_files):
        base_name = os.path.splitext(os.path.basename(gen_path))[0]  # SMILES_GRU_1_64_100_generated
        exp_name = base_name.replace("_generated", "")               # SMILES_GRU_1_64_100
        json_path = os.path.join(EVAL_DIR, f"{exp_name}_metrics.json")

        print(f"[{i+1}/{len(gen_files)}] {exp_name}", flush=True)

        # Si ya existe el JSON, cargar sin recalcular
        if os.path.exists(json_path):
            print(f"  Ya evaluado. Cargando desde JSON...", flush=True)
            with open(json_path, "r") as f:
                metrics = json.load(f)
        else:
            try:
                metrics = evaluate_single(gen_path, json_path)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                metrics = {}

        all_results.append((exp_name, metrics))
        print("", flush=True)

    # Generar resumen .txt
    summary_path = os.path.join(OUTPUTS_DIR, "eval_moses_resumen.txt")
    
    # Header
    header_cols = ["Experimento"] + [label for _, label in summary_keys]
    header_line = " & ".join(header_cols)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(header_line + "\n")
        
        for exp_name, metrics in all_results:
            values = [exp_name]
            for key, _ in summary_keys:
                val = metrics.get(key, None)
                values.append(format_value(val))
            
            line = " & ".join(values)
            f.write(line + "\n")
            print(line, flush=True)

    print(f"\nResumen guardado: {summary_path}", flush=True)
    print("=== Evaluación completada ===", flush=True)


if __name__ == "__main__":
    main()
