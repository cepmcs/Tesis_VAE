#!/bin/bash
#SBATCH --job-name=VAE_MOL
#SBATCH --output=salida_entrenamiento.out
#SBATCH --error=error_entrenamiento.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00

/etc/profile

source /home/cperez/miniconda3/bin/activate vae-mol

# --- Batería automática de pruebas (Límite 12 horas) ---
# Tienes que hacer diccionarios paralelos para DATA y NOMBRES
DATA_BATERIA=("data/mosesSMILES_processed.pt" "data/mosesSELFIES_processed.pt")
NOMBRES_BATERIA=("SMILES" "SELFIES")

EPOCHS_LIST=(100)              
LATENT_DIMS=(64 128 256)
NUM_LAYERS=(1 2)

# Variable para rastrear el lock activo y limpiarlo si el script muere
CURRENT_LOCK=""
cleanup() {
    if [ -n "$CURRENT_LOCK" ] && [ -f "$CURRENT_LOCK" ]; then
        echo "Señal recibida. Eliminando lock: $CURRENT_LOCK"
        rm -f "$CURRENT_LOCK"
    fi
    exit 1
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Un bucle iterando sobre el índice (0 o 1) para ambos arrays a la vez
for idx in "${!DATA_BATERIA[@]}"; do
    DATA="${DATA_BATERIA[$idx]}"
    REP_NAME="${NOMBRES_BATERIA[$idx]}"
    
    for EPOCH in "${EPOCHS_LIST[@]}"; do
        for LATENT in "${LATENT_DIMS[@]}"; do
            for LAYERS in "${NUM_LAYERS[@]}"; do
                
                # Nombre formato: SMILES_GRU_1_64_100
                EXP_NAME="${REP_NAME}_GRU_${LAYERS}_${LATENT}_${EPOCH}"
                MODEL_FILE="Tesis_VAE/models/${EXP_NAME}.pth"
                LOCK_FILE="Tesis_VAE/models/${EXP_NAME}.lock"
                
                echo "======================================================"
                # Control de reinicio: Si existe modelo o lock, se salta
                if [ -f "$MODEL_FILE" ]; then
                    echo "El modelo $EXP_NAME ya existe. Saltando al siguiente..."
                elif [ -f "$LOCK_FILE" ]; then
                    echo "El modelo $EXP_NAME está siendo entrenado por otro proceso. Saltando..."
                else
                    echo "CORRIENDO EXPERIMENTO: $EXP_NAME"
                    
                    # Crear lock file para evitar duplicados en paralelo
                    touch "$LOCK_FILE"
                    CURRENT_LOCK="$LOCK_FILE"
                    
                    # Ejecutando entrenamiento
                    python Tesis_VAE/src/train.py \
                        --latent_dim $LATENT \
                        --epochs $EPOCH \
                        --data_path $DATA \
                        --exp_name $EXP_NAME \
                        --num_layers $LAYERS
                    
                    # Eliminar lock al terminar
                    rm -f "$LOCK_FILE"
                    CURRENT_LOCK=""
                fi
                
            done
        done
    done
done

trap - EXIT
echo "¡BATERÍA COMPLETADA!"