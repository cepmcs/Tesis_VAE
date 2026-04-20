import pandas as pd
import selfies as sf
import torch
from tqdm import tqdm
import os
import urllib.request

# Directorio raíz del proyecto (un nivel arriba de scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config para MOSES
DATA_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
FILENAME = os.path.join(ROOT_DIR, "data", "moses.csv")
PROCESSED_FILE = os.path.join(ROOT_DIR, "data", "mosesSELFIES_processed.pt")
MAX_LEN = 65

def get_data():
    """Descarga el CSV de MOSES y retorna los splits oficiales train y test."""
    if not os.path.exists(FILENAME):
        print(f"Descargando datos de MOSES...")
        try:
            urllib.request.urlretrieve(DATA_URL, FILENAME)
        except Exception as e:
            print(f"Error descarga: {e}")
            exit()
    
    df = pd.read_csv(FILENAME)
    
    # Usar los splits OFICIALES de MOSES (columna SPLIT)
    df_train = df[df['SPLIT'] == 'train'].copy()
    df_test = df[df['SPLIT'] == 'test'].copy()
    
    # Limpiamos espacios
    df_train['SMILES'] = df_train['SMILES'].str.strip()
    df_test['SMILES'] = df_test['SMILES'].str.strip()
    
    return df_train['SMILES'].tolist(), df_test['SMILES'].tolist()

def smiles_to_selfies(smiles_list, desc_label=""):
    """Convierte SMILES a SELFIES, filtrando por longitud máxima."""
    selfies_list = []
    
    for sm in tqdm(smiles_list, desc=f"Convirtiendo a SELFIES ({desc_label})"):
        try:
            sel = sf.encoder(sm)
            if sel and len(list(sf.split_selfies(sel))) < MAX_LEN:
                selfies_list.append(sel)
        except:
            continue

    return selfies_list

def build_vocab(selfies_list):
    """Construye el vocabulario a partir de los datos de ENTRENAMIENTO únicamente."""
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
    
    return char_to_idx, idx_to_char

def encode_data(selfies_list, char_to_idx, desc_label=""):
    """Convierte datos SELFIES a tensor de índices usando el vocabulario dado."""
    sos_idx = char_to_idx['[SOS]']
    eos_idx = char_to_idx['[EOS]']
    pad_idx = char_to_idx['[PAD]']
    unk_idx = char_to_idx['[UNK]']
    
    data_indices = []
    for s in tqdm(selfies_list, desc=f"Codificando {desc_label}"):
        tokens = list(sf.split_selfies(s))
        indices = [char_to_idx.get(t, unk_idx) for t in tokens]
        
        # Añadir SOS al inicio y EOS al final
        indices = [sos_idx] + indices + [eos_idx]
        
        # Padding o truncado (MAX_LEN incluye SOS y EOS)
        if len(indices) < MAX_LEN:
            indices += [pad_idx] * (MAX_LEN - len(indices))
        else:
            indices = indices[:MAX_LEN-1] + [eos_idx]  # Asegurar EOS al final
        data_indices.append(indices)

    return torch.tensor(data_indices, dtype=torch.long)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    
    # 1. Obtener splits oficiales de MOSES
    train_smiles, test_smiles = get_data()
    print(f"MOSES oficial -> Train: {len(train_smiles)} | Test: {len(test_smiles)}")
    
    # 2. Convertir SMILES -> SELFIES para ambos splits
    train_selfies = smiles_to_selfies(train_smiles, desc_label="train")
    test_selfies = smiles_to_selfies(test_smiles, desc_label="test")
    print(f"Tras filtrar  -> Train: {len(train_selfies)} | Test: {len(test_selfies)}")
    
    # 3. Vocabulario se construye SOLO con datos de entrenamiento
    c2i, i2c = build_vocab(train_selfies)
    print(f"Vocabulario: {len(c2i)} tokens (construido solo con train)")
    
    # 4. Codificar ambos splits con el MISMO vocabulario
    train_tensor = encode_data(train_selfies, c2i, desc_label="train")
    test_tensor = encode_data(test_selfies, c2i, desc_label="test")
    
    # 5. Guardar con train y test separados
    torch.save({
        "train_data": train_tensor,
        "test_data": test_tensor,
        "vocab_stoi": c2i, 
        "vocab_itos": i2c,
        "max_len": MAX_LEN
    }, PROCESSED_FILE)
    
    print(f"\nDatos guardados en {PROCESSED_FILE}")
    print(f"  train_data: {train_tensor.shape}")
    print(f"  test_data:  {test_tensor.shape}")