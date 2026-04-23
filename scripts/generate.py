"""
Genera 50,000 moléculas por cada modelo .pth en models/.
Detecta si es GRU o LSTM del nombre y usa la arquitectura correcta.
Los modelos SELFIES se convierten a SMILES antes de guardar.
Salida: outputs/{nombre_modelo}_generated.txt (1 SMILES por línea)
"""
import torch
import torch.nn.functional as F
import os
import sys
import glob
import selfies as sf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from vae_model import MolecularVAE
from vae_model_lstm import MolecularVAE_LSTM

# --- CONFIGURACIÓN ---
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
NUM_MOLECULES = 50000
MAX_LEN = 100
TEMP = 1.0
BATCH_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """Carga modelo, detecta GRU/LSTM del nombre, devuelve modelo + vocab + tipo."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    vocab_stoi = checkpoint['vocab_stoi']
    vocab_itos = checkpoint['vocab_itos']
    hyper = checkpoint['hyperparams']
    
    model_name = os.path.basename(model_path).upper()
    is_lstm = "_LSTM_" in model_name
    
    if is_lstm:
        model = MolecularVAE_LSTM(
            vocab_size=hyper['vocab_size'],
            embed_size=hyper['embed'],
            hidden_size=hyper['hidden'],
            latent_size=hyper['latent'],
            num_layers=hyper.get('num_layers', 1)
        ).to(DEVICE)
    else:
        model = MolecularVAE(
            vocab_size=hyper['vocab_size'],
            embed_size=hyper['embed'],
            hidden_size=hyper['hidden'],
            latent_size=hyper['latent'],
            num_layers=hyper.get('num_layers', 1)
        ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, vocab_stoi, vocab_itos, hyper['latent'], is_lstm


def decode_gru(model, z, sos_idx, eos_idx):
    """Decodifica un batch de latentes z usando GRU."""
    batch_size = z.size(0)
    h = model.decoder_input(z).unsqueeze(0).repeat(model.num_layers, 1, 1)
    
    current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(DEVICE)
    decoded = []
    finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(MAX_LEN):
            embed = model.embedding(current_token)
            out, h = model.decoder_rnn(embed, h)
            logits = model.fc_out(out.squeeze(1))
            probs = F.softmax(logits / TEMP, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = torch.where(finished.unsqueeze(1), torch.zeros_like(next_token), next_token)
            decoded.append(next_token)
            current_token = next_token
            finished = finished | (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break
    
    return torch.cat(decoded, dim=1)


def decode_lstm(model, z, sos_idx, eos_idx):
    """Decodifica un batch de latentes z usando LSTM."""
    batch_size = z.size(0)
    h = model.decoder_input_h(z).unsqueeze(0).repeat(model.num_layers, 1, 1)
    c = model.decoder_input_c(z).unsqueeze(0).repeat(model.num_layers, 1, 1)
    
    current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(DEVICE)
    decoded = []
    finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(MAX_LEN):
            embed = model.embedding(current_token)
            out, (h, c) = model.decoder_rnn(embed, (h, c))
            logits = model.fc_out(out.squeeze(1))
            probs = F.softmax(logits / TEMP, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = torch.where(finished.unsqueeze(1), torch.zeros_like(next_token), next_token)
            decoded.append(next_token)
            current_token = next_token
            finished = finished | (next_token.squeeze(1) == eos_idx)
            if finished.all():
                break
    
    return torch.cat(decoded, dim=1)


def indices_to_strings(indices_tensor, vocab_itos):
    """Convierte tensor de índices a lista de strings."""
    results = []
    special = {'[PAD]', '[SOS]', '[EOS]', '[UNK]'}
    
    for seq in indices_tensor.cpu().numpy():
        tokens = []
        for idx in seq:
            if idx == 0:
                continue
            token = vocab_itos[idx]
            if token == '[EOS]':
                break
            if token not in special:
                tokens.append(token)
        results.append("".join(tokens))
    
    return results


def generate_for_model(model_path):
    """Genera 50k moléculas para un modelo y guarda el .txt"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_file = os.path.join(OUTPUTS_DIR, f"{model_name}_generated.txt")
    
    # Saltar si ya existe
    if os.path.exists(output_file):
        print(f"  Ya existe. Saltando.", flush=True)
        return
    
    # Cargar modelo
    model, stoi, itos, latent_dim, is_lstm = load_model(model_path)
    decode_fn = decode_lstm if is_lstm else decode_gru
    rnn_label = "LSTM" if is_lstm else "GRU"
    is_selfies = model_name.upper().startswith("SELFIES")
    
    print(f"  Tipo: {rnn_label} | Rep: {'SELFIES' if is_selfies else 'SMILES'}", flush=True)
    
    # Generar en lotes
    sos_idx = stoi['[SOS]']
    eos_idx = stoi['[EOS]']
    all_strings = []
    
    for i in range(0, NUM_MOLECULES, BATCH_SIZE):
        batch = min(BATCH_SIZE, NUM_MOLECULES - i)
        z = torch.randn(batch, latent_dim).to(DEVICE)
        indices = decode_fn(model, z, sos_idx, eos_idx)
        all_strings.extend(indices_to_strings(indices, itos))
        
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"    {len(all_strings)}/{NUM_MOLECULES}...", flush=True)
    
    # SELFIES → SMILES 
    if is_selfies:
        print(f"  Convirtiendo SELFIES → SMILES...", flush=True)
        smiles = []
        for sel in all_strings:
            try:
                sm = sf.decoder(sel)
                smiles.append(sm if sm else "")
            except:
                smiles.append("")
    else:
        smiles = all_strings
    
    # Guardar (1 SMILES por línea)
    with open(output_file, "w") as f:
        for sm in smiles:
            if sm.strip():
                f.write(sm.strip() + "\n")
    
    print(f"  Guardado: {output_file} ({len(smiles)} moléculas)", flush=True)


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    pth_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
    
    if not pth_files:
        print(f"No se encontraron modelos en {MODELS_DIR}")
        return
    
    print(f"Modelos: {len(pth_files)} | Moléculas por modelo: {NUM_MOLECULES} | Device: {DEVICE}", flush=True)
    print("", flush=True)
    
    for i, path in enumerate(pth_files):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[{i+1}/{len(pth_files)}] {name}", flush=True)
        try:
            generate_for_model(path)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
        print("", flush=True)
    
    print("Generación completada.", flush=True)


if __name__ == "__main__":
    main()