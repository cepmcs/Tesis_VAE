import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from vae_model import MolecularVAE, vae_loss_function
import sys
import argparse
import csv
import time


# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- config Argparse ---
parser = argparse.ArgumentParser(description="Entrenamiento de VAE")
parser.add_argument('--latent_dim', type=int, required=True, help="Dimensión latente a usar")
parser.add_argument('--epochs', type=int, required=True, help="Número de épocas del experimento")
parser.add_argument('--data_path', type=str, required=True, help="Ruta del dataset")
parser.add_argument('--exp_name', type=str, required=True, help="Nombre único para guardar el modelo")
parser.add_argument('--num_layers', type=int, default=1, help="Cantidad de capas GRU")
args = parser.parse_args()

# --- Configuración global ---
BATCH_SIZE = 128
EPOCHS = args.epochs
LEARNING_RATE = 1e-3
LATENT_DIM = args.latent_dim      
HIDDEN_DIM = LATENT_DIM
EMBED_DIM = 128       
KL_START = 0
KL_END = 0.3      
KL_ANNEAL_EPOCHS = int(EPOCHS * 0.25)

# Paths dinámicos
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(ROOT_DIR, args.data_path)
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", f"{args.exp_name}.pth")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "models", "checkpoints", f"{args.exp_name}_checkpoint.pth")

# --- Función de Entrenamiento ---
def train():
    start_time = time.time()
    print(f"--- Iniciando entrenamiento GRU en: {DEVICE} ---", flush=True)
    
    if LATENT_DIM >= 256:
        max_clip = 1.0
    else:
        max_clip = 2.0  # Bajado de 5.0 a 2.0 para mayor estabilidad en SELFIES
    
    # 1. Cargar Datos
    if not os.path.exists(DATA_PATH):
        print(f"No se encuentra {DATA_PATH}")

    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "outputs"), exist_ok=True)

    # Cargar datos preprocesados
    saved_data = torch.load(DATA_PATH)
    vocab_stoi = saved_data["vocab_stoi"]
    vocab_size = len(vocab_stoi)

    # Usar splits oficiales de MOSES directamente
    if "train_data" in saved_data:
        train_tensor = saved_data["train_data"]
        test_tensor = saved_data["test_data"]
    else:
        print("ERROR: Formato de datos antiguo. Re-ejecutar scripts/data_prep.py", flush=True)
        return

    print(f"Vocabulario: {vocab_size} tokens", flush=True)
    print(f"Train (MOSES train): {len(train_tensor)} | Val (MOSES test): {len(test_tensor)}", flush=True)

    # DataLoaders directos con los splits oficiales de MOSES
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=2, pin_memory=True, persistent_workers=True)
    
    print(f"Train: {len(train_tensor)} | Validación: {len(test_tensor)}", flush=True)

    # 2. Inicializar Modelo 
    model = MolecularVAE(
        vocab_size, 
        EMBED_DIM, 
        HIDDEN_DIM, 
        LATENT_DIM,
        num_layers=args.num_layers
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Listas para guardar historial de loss y accuracy
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [],
    }

    # Checkpoint: reanudar si existe
    start_epoch = 0
    elapsed_time = 0.0  # Tiempo acumulado de corridas anteriores (segundos)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Reanudando desde checkpoint: {CHECKPOINT_PATH}", flush=True)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt['history']
        elapsed_time = ckpt.get('elapsed_time', 0.0)
        print(f"Reanudando desde epoch {start_epoch + 1}/{EPOCHS} (tiempo previo: {elapsed_time/60:.1f} min)", flush=True)

    # 3. Bucle de Entrenamiento
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        # KL Annealing 
        if epoch < KL_ANNEAL_EPOCHS:
            kl_weight = KL_START + (KL_END - KL_START) * (epoch / KL_ANNEAL_EPOCHS)
        else:
            kl_weight = KL_END

        #  ENTRENAMIENTO 
        # Estadísticas por epoch    
        epoch_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]", mininterval=300, disable=not sys.stderr.isatty())
        
        for batch in progress:
            x = batch[0].to(DEVICE)
            
            # Forward 
            prediction, mu, logvar = model(x)
            
            # Calcular Error 
            loss, recon_loss, kl_loss = vae_loss_function(prediction, x, mu, logvar, kl_weight)
            
            # Backward 
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_clip) # Clipping de gradiente ajustado
            optimizer.step()
            
            # Acumular Loss 
            epoch_loss += loss.item()
            
            #  Cálculo de Accuracy de Reconstrucción
            # Índice de la letra con mayor probabilidad 
            pred_indices = prediction.argmax(dim=-1)
            
            # Target es x[:, 1:] (sin SOS)
            target = x[:, 1:]
            
            # Máscara para ignorar tokens de padding 
            mask = target != 0
            batch_tokens = mask.sum().item()
            matches = (pred_indices == target) & mask
            
            correct_tokens += matches.sum().item()
            total_tokens += batch_tokens
            
            # Actualizar barra
            progress.set_postfix({
                'Loss/token': loss.item() / batch_tokens,
                'KL': f"{kl_weight:.4f}"
            })

        # Estadísticas del Epoch de Train
        train_avg_loss = epoch_loss / total_tokens
        train_avg_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Guardar en historial 
        history['train_loss'].append(train_avg_loss)
        history['train_accuracy'].append(train_avg_acc)
        
        #  VALIDACIÓN 
        model.eval()
        val_epoch_loss = 0
        val_correct_tokens = 0
        val_total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                
                # Forward
                prediction, mu, logvar = model(x)
                
                # Calcular Loss
                loss, recon, kl = vae_loss_function(prediction, x, mu, logvar, kl_weight)
                val_epoch_loss += loss.item()
                
                # Calcular Accuracy
                pred_indices = prediction.argmax(dim=-1)
                target = x[:, 1:]  # Target sin SOS
                mask = target != 0
                batch_tokens = mask.sum().item()
                matches = (pred_indices == target) & mask
                
                val_correct_tokens += matches.sum().item()
                val_total_tokens += batch_tokens
        
        # Estadísticas de Validación
        val_avg_loss = val_epoch_loss / val_total_tokens if val_total_tokens > 0 else 0
        val_avg_acc = (val_correct_tokens / val_total_tokens) * 100 if val_total_tokens > 0 else 0
        
        # Guardar en historial
        history['val_loss'].append(val_avg_loss)
        history['val_accuracy'].append(val_avg_acc)
        
        # Volver a modo train
        model.train()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train - Loss: {train_avg_loss:.4f} | Acc: {train_avg_acc:.2f}%", flush=True)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val   - Loss: {val_avg_loss:.4f} | Acc: {val_avg_acc:.2f}%", flush=True)

        # Guardar checkpoint (se sobreescribe cada epoch)
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'history': history,
            'elapsed_time': elapsed_time + (time.time() - start_time),
        }, CHECKPOINT_PATH)

    # 4. Guardar modelo entrenado (Última época)
    print("Guardando modelo...", flush=True)
    torch.save({
        'model_state': model.state_dict(),
        'vocab_stoi': vocab_stoi,
        'vocab_itos': saved_data['vocab_itos'],
        'hyperparams': {
            'embed': EMBED_DIM, 
            'hidden': HIDDEN_DIM, 
            'latent': LATENT_DIM,
            'vocab_size': vocab_size,
            'num_layers': args.num_layers
        },
        'history': history,
        'epoch': EPOCHS
    }, MODEL_SAVE_PATH)
    print(f"Modelo guardado en {MODEL_SAVE_PATH}", flush=True)

    # Eliminar checkpoint tras guardar modelo final
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"Checkpoint eliminado: {CHECKPOINT_PATH}", flush=True)

    # 5. Guardar resultados en CSV global (Reemplaza a la gráfica)
    RESULTS_CSV = os.path.join(ROOT_DIR, "outputs", "fase1_resultados.csv")
    file_exists = os.path.isfile(RESULTS_CSV)
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    end_time = time.time()
    training_time_minutes = (elapsed_time + (end_time - start_time)) / 60.0
    
    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Experimento', 'Dataset', 'RNN_Type', 'Layers', 'Latent_Dim', 'Epochs', 'Final_Train_Loss', 'Final_Val_Loss', 'Final_Train_Acc', 'Final_Val_Acc', 'Time(min)'])
        
        writer.writerow([
            args.exp_name,
            args.data_path,
            'GRU', # Fijo para estos experimentos
            args.num_layers,
            LATENT_DIM, 
            EPOCHS,
            final_train_loss, 
            final_val_loss,
            round(final_train_acc, 2),
            round(final_val_acc, 2),
            round(training_time_minutes, 2)
        ])
        
    print(f"Resultados de la última época guardados en {RESULTS_CSV}")

if __name__ == "__main__":
    train()