import pandas as pd
import selfies as sf
import torch
from tqdm import tqdm
import os
import urllib.request
import moses

train = moses.get_dataset('train')
# config
DATA_URL = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
FILENAME = "zinc250k.csv"
PROCESSED_FILE = "data_processed.pt"
MAX_LEN = 100 

def get_data():
    if not os.path.exists(FILENAME):
        print(f"Descargando datos...")
        try:
            urllib.request.urlretrieve(DATA_URL, FILENAME)
        except Exception as e:
            print(f"Error descarga: {e}")
            exit()
    
    df = pd.read_csv(FILENAME)
    # Limpiamos espacios
    df['smiles'] = df['smiles'].str.strip()
    return df['smiles'].tolist()

def process_data(smiles_list):
    print(f"Procesando {len(smiles_list)} moléculas...")
    selfies_list = []
    
    # 1. SMILES -> SELFIES
    for sm in tqdm(smiles_list, desc="Convirtiendo a SELFIES"):
        try:
            sel = sf.encoder(sm)
            if sel and len(list(sf.split_selfies(sel))) < MAX_LEN:
                selfies_list.append(sel)
        except:
            continue

    # 2. Vocabulario
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    alphabet.add('[PAD]')
    alphabet.add('[UNK]')
    alphabet.add('[SOS]')
    alphabet.add('[EOS]')
    
    vocab = list(sorted(alphabet))
    if '[PAD]' in vocab:
        vocab.remove('[PAD]')
        vocab.insert(0, '[PAD]') # PAD en índice 0
        
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for i, c in enumerate(vocab)}
    
    print(f"Vocabulario: {len(vocab)} tokens")

    # 3. Tokenizar con SOS y EOS
    sos_idx = char_to_idx['[SOS]']
    eos_idx = char_to_idx['[EOS]']
    pad_idx = char_to_idx['[PAD]']
    
    data_indices = []
    for s in tqdm(selfies_list, desc="Tokenizando"):
        tokens = list(sf.split_selfies(s))
        indices = [char_to_idx.get(t, char_to_idx['[UNK]']) for t in tokens]
        
        # Añadir SOS al inicio y EOS al final
        indices = [sos_idx] + indices + [eos_idx]
        
        # Padding o truncado (MAX_LEN incluye SOS y EOS)
        if len(indices) < MAX_LEN:
            indices += [pad_idx] * (MAX_LEN - len(indices))
        else:
            indices = indices[:MAX_LEN-1] + [eos_idx]  # Asegurar EOS al final
        data_indices.append(indices)

    return torch.tensor(data_indices, dtype=torch.long), char_to_idx, idx_to_char

if __name__ == "__main__":
    smiles = get_data()
    tensor_data, c2i, i2c = process_data(smiles)
    
    torch.save({
        "data": tensor_data, "vocab_stoi": c2i, 
        "vocab_itos": i2c, "max_len": MAX_LEN
    }, PROCESSED_FILE)
    print(f"Datos guardados en {PROCESSED_FILE}")