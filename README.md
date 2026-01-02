# VAE para Generación de Moléculas

Implementación de un Variational Autoencoder (VAE) para la generación y modelado de moléculas utilizando representación SMILES.

## Descripción

Este proyecto implementa un VAE basado en redes recurrentes (GRU) para aprender representaciones latentes de moléculas del dataset ZINC250K y generar nuevas estructuras moleculares.

## Estructura del Proyecto

- `vae_model.py`: Definición del modelo VAE con encoder-decoder GRU
- `data_prep.py`: Preprocesamiento del dataset ZINC250K
- `train.py`: Script de entrenamiento con KL annealing
- `generate.py`: Generación de nuevas moléculas desde el espacio latente
- `latent_viz.py`: Visualización del espacio latente
- `zinc250k.csv`: Dataset de moléculas ZINC

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### 1. Preparar los datos
```bash
python data_prep.py
```

### 2. Entrenar el modelo
```bash
python train.py
```

### 3. Generar moléculas
```bash
python generate.py
```

### 4. Visualizar espacio latente
```bash
python latent_viz.py
```

## Configuración del Modelo

- Dimensión latente: 128
- Dimensión oculta: 128
- Dimensión de embedding: 128
- KL annealing: 0 a 0.05 en 20 epochs
- Batch size: 128

## Requisitos

- Python 3.8+
- PyTorch 2.1+
- RDKit
- SELFIES

## Resultados

El modelo entrenado se guarda en `vae_model_best.pth` y puede generar moléculas válidas en formato SMILES.
