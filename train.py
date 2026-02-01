import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os

# Importamos el modelo y la función de pérdida
from vae_model import MolecularVAE, vae_loss_function
from plot_utils import plot_training 

# --- config ---
BATCH_SIZE = 128       # Tamaño del batch
EPOCHS = 50           # Número total de epochs
LEARNING_RATE = 1e-3  # Tasa de aprendizaje

# Dimensiones 128/128/128
LATENT_DIM = 128      
HIDDEN_DIM = 128      
EMBED_DIM = 128       

# KL ANNEALING 
# Valores para el KL Annealing
KL_START = 0 
KL_END = 0.05       
KL_ANNEAL_EPOCHS = 20 # Número de epochs para hacer annealing

# Paths y dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data_processed.pt" 
MODEL_SAVE_PATH = "vae_model.pth" 
PLOT_FILE = "training_loss.png" # Nombre de la gráfica final

# --- Función de Entrenamiento ---
def train():
    print(f"--- Iniciando entrenamiento  en: {DEVICE} ---")
    
    # 1. Cargar Datos
    if not os.path.exists(DATA_PATH):
        print("No se encuentra data_processed.pt")
        return

    # Cargar datos preprocesados
    saved_data = torch.load(DATA_PATH)
    data_tensor = saved_data["data"]
    vocab_stoi = saved_data["vocab_stoi"]
    vocab_size = len(vocab_stoi)
    

    print(f"Vocabulario: {vocab_size} tokens")
    print(f"Muestras totales: {len(data_tensor)}")

    # Split 80% train / 20% validation
    dataset = TensorDataset(data_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    print(f"Train: {train_size} | Validación: {val_size}")

    # 2. Inicializar Modelo 
    model = MolecularVAE(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Listas para guardar historial de loss y accuracy
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [],
        'train_elbo': [], 'val_elbo': []  # ELBO por molécula
    }

    # 3. Bucle de Entrenamiento
    model.train()

    for epoch in range(EPOCHS):
        # KL Annealing 
        if epoch < KL_ANNEAL_EPOCHS:
            kl_weight = KL_START + (KL_END - KL_START) * (epoch / KL_ANNEAL_EPOCHS)
        else:
            kl_weight = KL_END

        # ========== ENTRENAMIENTO ==========
        # Estadísticas por epoch    
        epoch_loss = 0
        epoch_recon = 0  # Para ELBO
        epoch_kl = 0     # Para ELBO
        correct_tokens = 0
        total_tokens = 0
        num_samples = 0  # Contar moléculas para ELBO
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        
        for batch in progress:
            x = batch[0].to(DEVICE)
            
            # Forward 
            prediction, mu, logvar = model(x)
            
            # Calcular Error 
            loss, recon, kl = vae_loss_function(prediction, x, mu, logvar, kl_weight)
            
            # Backward 
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0) # Clipping de gradiente
            optimizer.step()
            
            # Acumular Loss 
            epoch_loss += loss.item()
            epoch_recon += recon.item()  # Para ELBO
            epoch_kl += kl.item()        # Para ELBO
            num_samples += x.size(0)     # Contar moléculas
            
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
        train_elbo = -(epoch_recon + epoch_kl) / num_samples  # ELBO por molécula
        
        # Guardar en historial 
        history['train_loss'].append(train_avg_loss)
        history['train_accuracy'].append(train_avg_acc)
        history['train_elbo'].append(train_elbo)
        
        # ========== VALIDACIÓN ==========
        model.eval()
        val_epoch_loss = 0
        val_epoch_recon = 0  # Para ELBO
        val_epoch_kl = 0     # Para ELBO
        val_correct_tokens = 0
        val_total_tokens = 0
        val_num_samples = 0  # Contar moléculas
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                
                # Forward
                prediction, mu, logvar = model(x)
                
                # Calcular Loss
                loss, recon, kl = vae_loss_function(prediction, x, mu, logvar, kl_weight)
                val_epoch_loss += loss.item()
                val_epoch_recon += recon.item()  # Para ELBO
                val_epoch_kl += kl.item()        # Para ELBO
                val_num_samples += x.size(0)     # Contar moléculas
                
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
        val_elbo = -(val_epoch_recon + val_epoch_kl) / val_num_samples  # ELBO por molécula
        
        # Guardar en historial
        history['val_loss'].append(val_avg_loss)
        history['val_accuracy'].append(val_avg_acc)
        history['val_elbo'].append(val_elbo)
        
        # Volver a modo train
        model.train()
        
        print(f"    Train - Loss: {train_avg_loss:.4f} | Acc: {train_avg_acc:.2f}% | ELBO: {train_elbo:.2f}")
        print(f"    Val   - Loss: {val_avg_loss:.4f} | Acc: {val_avg_acc:.2f}% | ELBO: {val_elbo:.2f}")

    # 4. Guardar modelo entrenado
    print("Guardando modelo...")
    torch.save({
        'model_state': model.state_dict(),
        'vocab_stoi': vocab_stoi,
        'vocab_itos': saved_data['vocab_itos'],
        'hyperparams': {
            'embed': EMBED_DIM, 
            'hidden': HIDDEN_DIM, 
            'latent': LATENT_DIM,
            'vocab_size': vocab_size
        },
        'history': history
    }, MODEL_SAVE_PATH)
    print(f"Modelo guardado en {MODEL_SAVE_PATH}")

    # 5. Generar Gráfica Automática de Entrenamiento
    plot_training(history, PLOT_FILE)

if __name__ == "__main__":
    train()