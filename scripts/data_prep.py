import re
import pandas as pd
import torch
from tqdm import tqdm
import os
import urllib.request

# Directorio raíz del proyecto (dos niveles arriba de src/scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# config para MOSES
DATA_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
FILENAME = os.path.join(ROOT_DIR, "data", "moses.csv")
PROCESSED_FILE = os.path.join(ROOT_DIR, "data", "mosesSMILES_processed.pt")
MAX_LEN = 60

def get_data():
    if not os.path.exists(FILENAME):
        print(f"Descargando datos de MOSES...")
        try:
            urllib.request.urlretrieve(DATA_URL, FILENAME)
        except Exception as e:
            print(f"Error descarga: {e}")
            exit()
    
    df = pd.read_csv(FILENAME)
    
    # MOSES usa la columna "SMILES" y "SPLIT". Vamos a usar solo el split de entrenamiento
    df_train = df[df['SPLIT'] == 'train'].copy()
    
    # Limpiamos espacios
    df_train['SMILES'] = df_train['SMILES'].str.strip()
    
    return df_train['SMILES'].tolist()

# Expresión regular estándar para separar tokens de SMILES
SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|<|\*|\$|\%[0-9]{2}|[0-9])")

def tokenize_smiles(smiles):
    """Tokeniza un string SMILES usando la expresión regular."""
    return [token for token in SMILES_REGEX.findall(smiles)]

def process_data(smiles_list):
    print(f"Procesando {len(smiles_list)} moléculas...")
    filtered_smiles = []
    tokenized_smiles = []
    
    # 1. Filtrar por longitud y tokenizar
    for sm in tqdm(smiles_list, desc="Filtrando y tokenizando SMILES"):
        tokens = tokenize_smiles(sm)
        # +2 porque consideraremos SOS y EOS más adelante
        if len(tokens) + 2 <= MAX_LEN: 
            filtered_smiles.append(sm)
            tokenized_smiles.append(tokens)

    # 2. Construir Vocabulario
    alphabet = set()
    for tokens in tokenized_smiles:
        alphabet.update(tokens)
        
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

    # 3. Convertir a índices con SOS y EOS
    sos_idx = char_to_idx['[SOS]']
    eos_idx = char_to_idx['[EOS]']
    pad_idx = char_to_idx['[PAD]']
    unk_idx = char_to_idx['[UNK]']
    
    data_indices = []
    for tokens in tqdm(tokenized_smiles, desc="Convirtiendo a índices"):
        indices = [char_to_idx.get(t, unk_idx) for t in tokens]
        
        # Añadir SOS al inicio y EOS al final
        indices = [sos_idx] + indices + [eos_idx]
        
        # Padding
        if len(indices) < MAX_LEN:
            indices += [pad_idx] * (MAX_LEN - len(indices))
            
        data_indices.append(indices)

    return torch.tensor(data_indices, dtype=torch.long), char_to_idx, idx_to_char

if __name__ == "__main__":
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    smiles = get_data()
    tensor_data, c2i, i2c = process_data(smiles)
    
    torch.save({
        "data": tensor_data, "vocab_stoi": c2i, 
        "vocab_itos": i2c, "max_len": MAX_LEN
    }, PROCESSED_FILE)
    print(f"Datos guardados en {PROCESSED_FILE}")