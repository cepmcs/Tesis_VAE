#!/bin/bash
#SBATCH --job-name=VAE_FASE2_GRU
#SBATCH --output=salida_fase2_gru.out
#SBATCH --error=error_fase2_gru.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00

/etc/profile

source /home/cperez/miniconda3/bin/activate vae-mol

# =================================================================
# FASE 2 - Optimización de Hiperparámetros (Modelos GRU)
# Modelo seleccionado: SM-G 2 256 (100 épocas)
# Cuadrícula: LR ∈ {1e-3, 1e-4} × Batch ∈ {64, 128, 192}
# =================================================================

# Parámetros del modelo SM-G 2 256
DATA="data/mosesSMILES_processed.pt"
REP="SMILES"
LAYERS=2
LATENT=256
EPOCHS=100

# Cuadrícula de hiperparámetros Fase 2
LR_LIST=(1e-3 1e-4)
BATCH_LIST=(64 128 192)

# Control de limpieza de lock al interrumpir
CURRENT_LOCK=""
cleanup() {
    if [ -n "$CURRENT_LOCK" ] && [ -f "$CURRENT_LOCK" ]; then
        echo "Señal recibida. Eliminando lock: $CURRENT_LOCK"
        rm -f "$CURRENT_LOCK"
    fi
    exit 1
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT

for LR in "${LR_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do

        # Convertir LR a sufijo legible: 1e-3 -> lr1e3, 1e-4 -> lr1e4
        LR_SUFFIX=$(echo "$LR" | tr -d '.' | sed 's/e-/e/')

        # Nombre formato: SMILES_GRU_2_256_100_lr1e3_b64
        EXP_NAME="${REP}_GRU_${LAYERS}_${LATENT}_${EPOCHS}_lr${LR_SUFFIX}_b${BATCH}"
        MODEL_FILE="Tesis_VAE/models/${EXP_NAME}.pth"
        LOCK_FILE="Tesis_VAE/models/${EXP_NAME}.lock"

        echo "======================================================"
        if [ -f "$MODEL_FILE" ]; then
            echo "El modelo $EXP_NAME ya existe. Saltando..."
        elif [ -f "$LOCK_FILE" ]; then
            echo "El modelo $EXP_NAME está siendo entrenado. Saltando..."
        else
            echo "CORRIENDO EXPERIMENTO: $EXP_NAME"
            touch "$LOCK_FILE"
            CURRENT_LOCK="$LOCK_FILE"

            python Tesis_VAE/src/train.py \
                --latent_dim $LATENT \
                --epochs $EPOCHS \
                --data_path $DATA \
                --exp_name $EXP_NAME \
                --num_layers $LAYERS \
                --lr $LR \
                --batch_size $BATCH

            rm -f "$LOCK_FILE"
            CURRENT_LOCK=""
        fi

    done
done

trap - EXIT
echo "¡BATERÍA FASE 2 GRU COMPLETADA!"
