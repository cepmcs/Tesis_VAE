#!/bin/bash
#SBATCH --job-name=SEL_L2_D128
#SBATCH --output=salida_SEL_L2_D128.out
#SBATCH --error=error_SEL_L2_D128.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00

/etc/profile
source /home/cperez/miniconda3/bin/activate vae-mol

DATA="data/mosesSELFIES_processed.pt"
REP_NAME="SELFIES"
LATENT=128
LAYERS=2
EPOCHS_LIST=(100)

for EPOCH in "${EPOCHS_LIST[@]}"; do
    EXP_NAME="${REP_NAME}_GRU_${LAYERS}_${LATENT}_${EPOCH}"
    MODEL_FILE="Tesis_VAE/models/${EXP_NAME}.pth"
    
    echo "======================================================"
    if [ -f "$MODEL_FILE" ]; then
        echo "El modelo $EXP_NAME ya existe. Saltando..."
    else
        echo "CORRIENDO EXPERIMENTO: $EXP_NAME"
        python Tesis_VAE/src/train.py \
            --latent_dim $LATENT \
            --epochs $EPOCH \
            --data_path $DATA \
            --exp_name $EXP_NAME \
            --num_layers $LAYERS
    fi
done
echo "¡COMPLETADO $EXP_NAME!"
