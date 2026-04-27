#!/bin/bash
#SBATCH --job-name=VAE_FASE2_LSTM
#SBATCH --output=salida_fase2_lstm.out
#SBATCH --error=error_fase2_lstm.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=48:00:00

/etc/profile

source /home/cperez/miniconda3/bin/activate vae-mol

# =================================================================
# FASE 2 - Optimización de Hiperparámetros (Modelos LSTM)
# Modelos seleccionados: SM-L 2 256, SE-L 2 256, SM-L 1 256
# Cuadrícula: LR ∈ {1e-3, 1e-4} × Batch ∈ {64, 128, 192}
# =================================================================

# Definición de modelos base (DATA, NOMBRE_REP, NUM_LAYERS, LATENT, EPOCHS)
declare -A MODEL_DATA=(
    ["SM_L_2_256"]="data/mosesSMILES_processed.pt"
    ["SE_L_2_256"]="data/mosesSELFIES_processed.pt"
    ["SM_L_1_256"]="data/mosesSMILES_processed.pt"
)
declare -A MODEL_REP=(
    ["SM_L_2_256"]="SMILES"
    ["SE_L_2_256"]="SELFIES"
    ["SM_L_1_256"]="SMILES"
)
declare -A MODEL_LAYERS=(
    ["SM_L_2_256"]="2"
    ["SE_L_2_256"]="2"
    ["SM_L_1_256"]="1"
)
declare -A MODEL_LATENT=(
    ["SM_L_2_256"]="256"
    ["SE_L_2_256"]="256"
    ["SM_L_1_256"]="256"
)
declare -A MODEL_EPOCHS=(
    ["SM_L_2_256"]="300"
    ["SE_L_2_256"]="100"
    ["SM_L_1_256"]="300"
)

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

for MODEL_KEY in "SM_L_2_256" "SE_L_2_256" "SM_L_1_256"; do
    DATA="${MODEL_DATA[$MODEL_KEY]}"
    REP="${MODEL_REP[$MODEL_KEY]}"
    LAYERS="${MODEL_LAYERS[$MODEL_KEY]}"
    LATENT="${MODEL_LATENT[$MODEL_KEY]}"
    EPOCHS="${MODEL_EPOCHS[$MODEL_KEY]}"

    for LR in "${LR_LIST[@]}"; do
        for BATCH in "${BATCH_LIST[@]}"; do

            # Convertir LR a sufijo legible: 1e-3 -> lr1e3, 1e-4 -> lr1e4
            LR_SUFFIX=$(echo "$LR" | tr -d '.' | sed 's/e-/e/')

            # Nombre formato: SMILES_LSTM_2_256_300_lr1e3_b64
            EXP_NAME="${REP}_LSTM_${LAYERS}_${LATENT}_${EPOCHS}_lr${LR_SUFFIX}_b${BATCH}"
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

                python Tesis_VAE/src/train_lstm.py \
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
done

trap - EXIT
echo "¡BATERÍA FASE 2 LSTM COMPLETADA!"
