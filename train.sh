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
                
                echo "======================================================"
                # Control de reinicio: Si existe, se salta
                if [ -f "$MODEL_FILE" ]; then
                    echo "El modelo $EXP_NAME ya existe. Saltando al siguiente..."
                else
                    echo "CORRIENDO EXPERIMENTO: $EXP_NAME"
                    
                    # Ejecutando entrenamiento
                    python Tesis_VAE/src/train.py \
                        --latent_dim $LATENT \
                        --epochs $EPOCH \
                        --data_path $DATA \
                        --exp_name $EXP_NAME \
                        --num_layers $LAYERS
                fi
                
            done
        done
    done
done

echo "¡BATERÍA COMPLETADA!"