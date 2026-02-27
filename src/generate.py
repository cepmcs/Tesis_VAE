import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import os
import sys

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar tu arquitectura definida en vae_model.py
from vae_model import MolecularVAE

# Desactivar warnings de RDKit para moléculas inválidas durante la generación
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "vae_model.pth")
NUM_MOLECULES = 30000   # MOSES recomienda al menos 30k
MAX_LEN = 100          
TEMP = 1.0             # Temperatura para muestreo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_IMG = os.path.join(ROOT_DIR, "outputs", "generated_molecules.png")
OUTPUT_SMILES = os.path.join(ROOT_DIR, "outputs", "generated_smiles.txt")

def load_model_and_vocab():
    # Cargar el modelo entrenado y los vocabularios
    if not torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(MODEL_PATH)
        
    vocab_stoi = checkpoint['vocab_stoi']
    vocab_itos = checkpoint['vocab_itos']
    hyper = checkpoint['hyperparams']
    
    model = MolecularVAE(
        vocab_size=hyper['vocab_size'],
        embed_size=hyper['embed'],
        hidden_size=hyper['hidden'],
        latent_size=hyper['latent']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, vocab_stoi, vocab_itos, hyper['latent']

def decode_latent(model, z, vocab_stoi, vocab_itos, max_len=100, temp=1.0):
    # Decodificar latentes a secuencias de índices
    batch_size = z.size(0)
    h = model.decoder_input(z).unsqueeze(0)
    
    # Empezar con el token SOS
    sos_idx = vocab_stoi['[SOS]']
    eos_idx = vocab_stoi['[EOS]']
    
    current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(DEVICE)
    decoded_indices = []
    
    # Rastrear qué secuencias han terminado (generado EOS)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_len):
            embed = model.embedding(current_token)
            out, h = model.decoder_rnn(embed, h)
            logits = model.fc_out(out.squeeze(1))
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Si ya terminó, mantener PAD (0)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.zeros_like(next_token),
                next_token
            )
            
            decoded_indices.append(next_token)
            current_token = next_token
            
            # Marcar secuencias que generaron EOS
            finished = finished | (next_token.squeeze(1) == eos_idx)
            
            # Si todas terminaron, salir
            if finished.all():
                break
            
    decoded_indices = torch.cat(decoded_indices, dim=1)
    return decoded_indices

def indices_to_smiles(indices_tensor, vocab_itos):
    # Convertir secuencias de índices a cadenas SMILES usando SELFIES
    smiles_list = []
    indices_cpu = indices_tensor.cpu().numpy()
    
    # Tokens especiales a ignorar
    special_tokens = {'[PAD]', '[SOS]', '[EOS]', '[UNK]'}
    
    for seq in indices_cpu:
        tokens = []
        for idx in seq:
            if idx == 0:  # PAD
                continue
            token = vocab_itos[idx]
            if token == '[EOS]':  # Fin de secuencia
                break
            if token not in special_tokens:
                tokens.append(token)
        
        selfies_str = "".join(tokens)
        try:
            sm = sf.decoder(selfies_str)
            smiles_list.append(sm)
        except:
            smiles_list.append(None)
            
    return smiles_list

def main():
    print(f"--- Generando moléculas con {MODEL_PATH} ---")
    
    os.makedirs(os.path.dirname(OUTPUT_SMILES), exist_ok=True)

    # 1. Cargar modelo
    model, stoi, itos, latent_dim = load_model_and_vocab()
    
    # 2. Muestrear latentes (Prior Normal)
    # Generamos en lotes para no saturar memoria
    BATCH_SIZE = 500
    all_smiles = []
    
    for i in range(0, NUM_MOLECULES, BATCH_SIZE):
        batch = min(BATCH_SIZE, NUM_MOLECULES - i)
        z = torch.randn(batch, latent_dim).to(DEVICE)
        indices = decode_latent(model, z, stoi, itos, MAX_LEN, TEMP)
        batch_smiles = indices_to_smiles(indices, itos)
        all_smiles.extend(batch_smiles)
        print(f"  Generadas {len(all_smiles)}/{NUM_MOLECULES}...")
    
    # 3. Filtrar válidas
    valid_smiles = []
    valid_mols = []
    legends = []
    
    print("\n--- Filtrando moléculas válidas ---")
    for i, sm in enumerate(all_smiles):
        if sm:
            mol = Chem.MolFromSmiles(sm)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical)
                valid_mols.append(mol)
                legends.append(f"Mol {i}")

    print(f"\nTotal válidas: {len(valid_smiles)} / {NUM_MOLECULES}")

    # 4. GUARDAR SMILES PARA MOSES (1 por línea)
    with open(OUTPUT_SMILES, "w") as f:
        for sm in valid_smiles:
            f.write(sm + "\n")
    print(f"SMILES guardadas en: {OUTPUT_SMILES}")

    # 5. GENERAR IMAGEN CON RDKIT (primeras 50)
    if valid_mols:
        print(f"Generando imagen: {OUTPUT_IMG}...")
        img = Draw.MolsToGridImage(
            valid_mols[:50],           # Máximo 50 moléculas en la imagen
            molsPerRow=5,              # 5 columnas
            subImgSize=(300, 300),     # Tamaño de cada celda
            legends=legends[:50]       # Etiquetas
        )
        img.save(OUTPUT_IMG)
    else:
        print("No se generaron moléculas válidas")

if __name__ == "__main__":
    main()