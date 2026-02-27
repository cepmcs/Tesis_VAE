# VAE para Generación de Moléculas

Implementación de un Variational Autoencoder (VAE) para la generación y modelado de moléculas utilizando representación SMILES.

## Descripción

Este proyecto implementa un VAE basado en redes recurrentes (GRU) para aprender representaciones latentes de moléculas del dataset ZINC250K y generar nuevas estructuras moleculares.

## Estructura del Proyecto

```
Tesis_VAE/
├── data/                  # Datos crudos y procesados
│   ├── zinc250k.csv
│   └── data_processed.pt
├── models/                # Checkpoints del modelo
│   └── vae_model.pth
├── outputs/               # Resultados generados
│   ├── generated_smiles.txt
│   ├── metrics.json
│   └── *.png
├── src/                   # Código fuente
│   ├── vae_model.py       # Definición del modelo VAE (encoder-decoder GRU)
│   ├── data_prep.py       # Preprocesamiento del dataset ZINC250K
│   ├── train.py           # Script de entrenamiento con KL annealing
│   ├── generate.py        # Generación de nuevas moléculas
│   ├── eval_moses.py      # Evaluación con métricas MOSES
│   ├── analisis_resultados.py  # Análisis comparativo de propiedades
│   ├── latent_viz.py      # Visualización del espacio latente
│   └── plot_utils.py      # Utilidades de graficación
├── README.md
├── requirements.txt
└── .gitignore
```

## Instalación

    pip install -r requirements.txt

## Uso

Todos los scripts se ejecutan **desde la raíz del proyecto**:

### 1. Preparar los datos

    python src/data_prep.py

### 2. Entrenar el modelo

    python src/train.py

### 3. Generar moléculas

    python src/generate.py

### 4. Evaluar con MOSES

    conda run -n moses_fork python src/eval_moses.py

### 5. Análisis comparativo

    python src/analisis_resultados.py

### 6. Visualizar espacio latente

    python src/latent_viz.py

## Configuración del Modelo

- Dimensión latente: 128
- Dimensión oculta: 128
- Dimensión de embedding: 128
- KL annealing: 0 a 0.3 en 20 epochs
- Batch size: 128

## Requisitos

- Python 3.8+
- PyTorch 2.1+
- RDKit
- SELFIES

## Resultados

El modelo entrenado se guarda en `models/vae_model.pth` y puede generar moléculas válidas en formato SMILES.
