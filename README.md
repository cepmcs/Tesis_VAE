# VAE para Generación de Moléculas

VAE basado en GRU para aprender representaciones latentes de moléculas del dataset ZINC250K y generar nuevas estructuras moleculares.

## Estructura

- `src/` — Código fuente (modelo, entrenamiento, generación, evaluación)
- `data/` — Datos crudos y procesados
- `models/` — Checkpoints del modelo
- `outputs/` — SMILES generados, métricas y gráficos

## Uso

    pip install -r requirements.txt

Ejecutar desde la raíz del proyecto:

    python src/data_prep.py           # 1. Preparar datos
    python src/train.py               # 2. Entrenar
    python src/generate.py            # 3. Generar moléculas
    conda run -n moses_fork python src/eval_moses.py  # 4. Evaluar (MOSES)
    python src/analisis_resultados.py # 5. Análisis comparativo
    python src/latent_viz.py          # 6. Visualizar espacio latente
