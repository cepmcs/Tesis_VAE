import pandas as pd
import selfies as sf
from tqdm import tqdm
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(ROOT_DIR, "data", "moses.csv")

def main():
    if not os.path.exists(FILENAME):
        print(f"Error: No se encontró {FILENAME}.")
        return

    print("Cargando datos de MOSES...")
    df = pd.read_csv(FILENAME)
    
    df_train = df[df['SPLIT'] == 'train'].copy()
    smiles_list = df_train['SMILES'].str.strip().tolist()
    
    print(f"Procesando {len(smiles_list)} moléculas...")
    
    max_smiles_len = 0
    max_selfies_len = 0
    
    for sm in tqdm(smiles_list, desc="Analizando longitudes"):
        # Longitud SMILES (caracteres)
        if len(sm) > max_smiles_len:
            max_smiles_len = len(sm)
            
        # Longitud SELFIES (tokens)
        try:
            sel = sf.encoder(sm)
            if sel:
                sel_len = len(list(sf.split_selfies(sel)))
                if sel_len > max_selfies_len:
                    max_selfies_len = sel_len
        except:
            continue
            
    print("\n--- RESULTADOS ---")
    print(f"Longitud máxima en SMILES (caracteres): {max_smiles_len}")
    print(f"Longitud máxima en SELFIES (tokens): {max_selfies_len}")
    print("------------------")
    print(f"Al fijar MAX_LEN en data_prep.py, recuerda sumar 2 para los tokens [SOS] y [EOS].")

if __name__ == "__main__":
    main()
