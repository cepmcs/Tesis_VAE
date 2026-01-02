import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger

# Importar tu arquitectura definida en vae_model.py
from vae_model import MolecularVAE

# Desactivar warnings de RDKit para moléculas inválidas durante la generación
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACIÓN ---
MODEL_PATH = "vae_model.pth"
NUM_MOLECULES = 10     # Generaremos 10
MAX_LEN = 100          
TEMP = 1.0           # Temperatura para muestreo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_IMG = "generated_molecules.png"

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
    current_token = torch.zeros(batch_size, 1, dtype=torch.long).to(DEVICE)
    decoded_indices = []
    
    with torch.no_grad():
        for _ in range(max_len):
            embed = model.embedding(current_token)
            out, h = model.decoder_rnn(embed, h)
            logits = model.fc_out(out.squeeze(1))
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            decoded_indices.append(next_token)
            current_token = next_token
            
    decoded_indices = torch.cat(decoded_indices, dim=1)
    return decoded_indices

def indices_to_smiles(indices_tensor, vocab_itos):
    # Convertir secuencias de índices a cadenas SMILES usando SELFIES
    smiles_list = []
    indices_cpu = indices_tensor.cpu().numpy()
    
    for seq in indices_cpu:
        tokens = []
        for idx in seq:
            if idx == 0: continue
            token = vocab_itos[idx]
            if token in ['[EOS]', '[PAD]']:
                break
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
    
    # 1. Cargar modelo
    model, stoi, itos, latent_dim = load_model_and_vocab()
    
    # 2. Muestrear latentes (Prior Normal)
    z = torch.randn(NUM_MOLECULES, latent_dim).to(DEVICE)
    
    # 3. Decodificar
    indices = decode_latent(model, z, stoi, itos, MAX_LEN, TEMP)
    generated_smiles = indices_to_smiles(indices, itos)
    
    # 4. Filtrar válidas
    valid_mols = []
    legends = []
    
    print("\n--- Procesando estructuras ---")
    for i, sm in enumerate(generated_smiles):
        if sm:
            mol = Chem.MolFromSmiles(sm)
            if mol:
                valid_mols.append(mol)
                legends.append(f"Mol {i}")
                print(f"[{i}] Generada: {sm}")

    print(f"\nTotal válidas: {len(valid_mols)} / {NUM_MOLECULES}")

    # 5. GENERAR IMAGEN CON RDKIT
    if valid_mols:
        print(f"{OUTPUT_IMG}...")
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