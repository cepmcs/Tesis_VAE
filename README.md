# VAE para Generación de Moléculas

VAE basado en GRU y LSTM para aprender representaciones latentes de moléculas del dataset MOSES y generar nuevas estructuras moleculares.

## Estructura

```
data/                 moses.csv y datos procesados (no versionado)
src/                  modelos (vae_model*.py) y entrenamiento (train*.py)
scripts/              preparación de datos, generación, evaluación y figuras
modelos_entrenados/   .pth por fase (no versionado)
resultados/           métricas, CSV y figuras por fase
```

## Uso

```bash
# 1. Datos (splits oficiales de MOSES)
python scripts/data_prep.py            # SMILES
python scripts/data_prepSelfies.py     # SELFIES

# 2. Entrenamiento
python src/train.py --latent_dim 256 --epochs 300 --data_path data/mosesSMILES_processed.pt \
    --exp_name SMILES_GRU_2_256_300 --num_layers 2 --fase fase2
python src/train_lstm.py ...           # mismos argumentos, modelo LSTM

# 3. Generación y evaluación MOSES
python scripts/generate.py --fase fase2
conda run -n moses_fork python scripts/eval_moses.py --fase fase2

# 4. Figuras del modelo final
python scripts/pca_latente.py
python scripts/plot_distribuciones.py
```

`--lr` (default 1e-3) y `--batch_size` (default 128) son opcionales.

## Experimentos

| Fase | Exploración |
|---|---|
| `fase1_100ep`, `fase1_300ep` | GRU y LSTM, SMILES y SELFIES, latente 64/128/256, 1 y 2 capas (lr 1e-3, batch 128) |
| `fase2` | lr {1e-3, 1e-4} × batch {64, 128, 192} sobre los mejores modelos de la fase 1 |

Modelo final: `SMILES_LSTM_2_256_300_lr1e4_b64` (resultados en `resultados/modelo_final/`).

## Datos

Se usan los splits de MOSES: train (1.584.663), test (176.074) y test_scaffolds (176.225). Durante el entrenamiento el test se usa para monitorear la pérdida; las métricas finales se calculan con `moses.get_all_metrics`.

## Dependencias

`requirements.txt`. La evaluación requiere MOSES (entorno `moses_fork`).
