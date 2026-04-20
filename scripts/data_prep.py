import re
import pandas as pd
import torch
from tqdm import tqdm
import os
import urllib.request

# Directorio raíz del proyecto (un nivel arriba de scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# config para MOSES
DATA_URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
FILENAME = os.path.join(ROOT_DIR, "data", "moses.csv")
PROCESSED_FILE = os.path.join(ROOT_DIR, "data", "mosesSMILES_processed.pt")
MAX_LEN = 60

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

# Expresión regular estándar para separar tokens de SMILES
SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|\=|#|-|\+|\\\\|\/|:|~|@|\?|>|<|\*|\$|\%[0-9]{2}|[0-9])")

def tokenize_smiles(smiles):
    """Tokeniza un string SMILES usando la expresión regular."""
    return [token for token in SMILES_REGEX.findall(smiles)]

def filter_and_tokenize(smiles_list, desc_label=""):
    """Filtra por longitud máxima y tokeniza una lista de SMILES."""
    filtered_smiles = []
    tokenized_smiles = []
    
    for sm in tqdm(smiles_list, desc=f"Filtrando y tokenizando {desc_label}"):
        tokens = tokenize_smiles(sm)
        # +2 porque consideraremos SOS y EOS más adelante
        if len(tokens) + 2 <= MAX_LEN: 
            filtered_smiles.append(sm)
            tokenized_smiles.append(tokens)

    return filtered_smiles, tokenized_smiles

def build_vocab(tokenized_smiles):
    """Construye el vocabulario a partir de los datos de ENTRENAMIENTO únicamente."""
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
    
    return char_to_idx, idx_to_char

def encode_data(tokenized_smiles, char_to_idx, desc_label=""):
    """Convierte datos tokenizados a tensor de índices usando el vocabulario dado."""
    sos_idx = char_to_idx['[SOS]']
    eos_idx = char_to_idx['[EOS]']
    pad_idx = char_to_idx['[PAD]']
    unk_idx = char_to_idx['[UNK]']
    
    data_indices = []
    for tokens in tqdm(tokenized_smiles, desc=f"Codificando {desc_label}"):
        indices = [char_to_idx.get(t, unk_idx) for t in tokens]
        
        # Añadir SOS al inicio y EOS al final
        indices = [sos_idx] + indices + [eos_idx]
        
        # Padding
        if len(indices) < MAX_LEN:
            indices += [pad_idx] * (MAX_LEN - len(indices))
            
        data_indices.append(indices)

    return torch.tensor(data_indices, dtype=torch.long)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    
    # 1. Obtener splits oficiales de MOSES
    train_smiles, test_smiles = get_data()
    print(f"MOSES oficial -> Train: {len(train_smiles)} | Test: {len(test_smiles)}")
    
    # 2. Filtrar y tokenizar ambos splits
    _, train_tokenized = filter_and_tokenize(train_smiles, desc_label="train")
    _, test_tokenized = filter_and_tokenize(test_smiles, desc_label="test")
    print(f"Tras filtrar  -> Train: {len(train_tokenized)} | Test: {len(test_tokenized)}")
    
    # 3. Vocabulario se construye SOLO con datos de entrenamiento
    c2i, i2c = build_vocab(train_tokenized)
    print(f"Vocabulario: {len(c2i)} tokens (construido solo con train)")
    
    # 4. Codificar ambos splits con el MISMO vocabulario
    train_tensor = encode_data(train_tokenized, c2i, desc_label="train")
    test_tensor = encode_data(test_tokenized, c2i, desc_label="test")
    
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