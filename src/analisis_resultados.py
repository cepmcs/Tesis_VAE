import torch
import torch.nn.functional as F
import selfies as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import os
import sys

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar arquitectura del VAE
from vae_model import MolecularVAE

# Desactivar warnings de RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "vae_model.pth")
ZINC_PATH = os.path.join(ROOT_DIR, "data", "zinc250k.csv")
NUM_GENERATED = 30000      # Número de moléculas a generar
MAX_LEN = 100
TEMP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PLOT = os.path.join(ROOT_DIR, "outputs", "comparacion_propiedades.png")


def load_model_and_vocab():
    """Carga el modelo VAE entrenado y los vocabularios."""
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
    """Decodifica vectores latentes a secuencias de índices."""
    batch_size = z.size(0)
    h = model.decoder_input(z).unsqueeze(0)
    
    sos_idx = vocab_stoi['[SOS]']
    eos_idx = vocab_stoi['[EOS]']
    
    current_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(DEVICE)
    decoded_indices = []
    finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_len):
            embed = model.embedding(current_token)
            out, h = model.decoder_rnn(embed, h)
            logits = model.fc_out(out.squeeze(1))
            probs = F.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.zeros_like(next_token),
                next_token
            )
            
            decoded_indices.append(next_token)
            current_token = next_token
            finished = finished | (next_token.squeeze(1) == eos_idx)
            
            if finished.all():
                break
            
    decoded_indices = torch.cat(decoded_indices, dim=1)
    return decoded_indices


def indices_to_smiles(indices_tensor, vocab_itos):
    """Convierte secuencias de índices a SMILES usando SELFIES."""
    smiles_list = []
    indices_cpu = indices_tensor.cpu().numpy()
    special_tokens = {'[PAD]', '[SOS]', '[EOS]', '[UNK]'}
    
    for seq in indices_cpu:
        tokens = []
        for idx in seq:
            if idx == 0:
                continue
            token = vocab_itos[idx]
            if token == '[EOS]':
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


def generate_molecules(model, vocab_stoi, vocab_itos, latent_dim, num_molecules, batch_size=500):
    """Genera moléculas muestreando del espacio latente."""
    print(f"Generando {num_molecules} moléculas...")
    
    all_smiles = []
    num_batches = (num_molecules + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch = min(batch_size, num_molecules - i * batch_size)
        z = torch.randn(current_batch, latent_dim).to(DEVICE)
        indices = decode_latent(model, z, vocab_stoi, vocab_itos, MAX_LEN, TEMP)
        smiles = indices_to_smiles(indices, vocab_itos)
        all_smiles.extend(smiles)
        
        if (i + 1) % 5 == 0 or i == num_batches - 1:
            print(f"  Procesados {min((i+1) * batch_size, num_molecules)} / {num_molecules}")
    
    return all_smiles


def calculate_properties(smiles_list, source_name=""):
    """
    Calcula Peso Molecular (MW) y LogP para una lista de SMILES.
    Retorna un DataFrame con las propiedades.
    """
    mw_list = []
    logp_list = []
    valid_smiles = []
    
    for sm in smiles_list:
        if sm is None:
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                mw_list.append(mw)
                logp_list.append(logp)
                valid_smiles.append(sm)
            except:
                continue
    
    df = pd.DataFrame({
        'SMILES': valid_smiles,
        'MW': mw_list,
        'LogP': logp_list,
        'Source': source_name
    })
    
    return df


def load_zinc_dataset(path, sample_size=None):
    """Carga el dataset ZINC250k y extrae SMILES."""
    print(f"Cargando dataset ZINC250k desde {path}...")
    df = pd.read_csv(path)
    
    # Limpiar SMILES (quitar saltos de línea)
    df['smiles'] = df['smiles'].str.strip()
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Muestra de {sample_size} moléculas seleccionada")
    
    return df['smiles'].tolist()


def plot_comparison(df_original, df_generated, output_path):
    """Genera gráficos KDE comparativos de MW y LogP."""
    
    # Combinar dataframes
    df_combined = pd.concat([df_original, df_generated], ignore_index=True)
    
    # Configurar estilo
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colores
    colors = {'ZINC250k (Original)': '#2E86AB', 'Generadas (VAE)': '#E94F37'}
    
    # --- Gráfico 1: Distribución de LogP ---
    ax1 = axes[0]
    for source in ['ZINC250k (Original)', 'Generadas (VAE)']:
        data = df_combined[df_combined['Source'] == source]['LogP']
        sns.kdeplot(
            data=data,
            ax=ax1,
            label=source,
            color=colors[source],
            linewidth=2,
            fill=True,
            alpha=0.3
        )
    
    ax1.set_xlabel('LogP', fontsize=12)
    ax1.set_ylabel('Densidad', fontsize=12)
    ax1.set_title('Distribución de LogP', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.axvline(x=df_original['LogP'].mean(), color=colors['ZINC250k (Original)'], 
                linestyle='--', alpha=0.7, label=f"Media Original: {df_original['LogP'].mean():.2f}")
    ax1.axvline(x=df_generated['LogP'].mean(), color=colors['Generadas (VAE)'], 
                linestyle='--', alpha=0.7, label=f"Media Generadas: {df_generated['LogP'].mean():.2f}")
    
    # --- Gráfico 2: Distribución de Peso Molecular ---
    ax2 = axes[1]
    for source in ['ZINC250k (Original)', 'Generadas (VAE)']:
        data = df_combined[df_combined['Source'] == source]['MW']
        sns.kdeplot(
            data=data,
            ax=ax2,
            label=source,
            color=colors[source],
            linewidth=2,
            fill=True,
            alpha=0.3
        )
    
    ax2.set_xlabel('Peso Molecular (Da)', fontsize=12)
    ax2.set_ylabel('Densidad', fontsize=12)
    ax2.set_title('Distribución de Peso Molecular', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axvline(x=df_original['MW'].mean(), color=colors['ZINC250k (Original)'], 
                linestyle='--', alpha=0.7)
    ax2.axvline(x=df_generated['MW'].mean(), color=colors['Generadas (VAE)'], 
                linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")


def print_statistics(df_original, df_generated):
    """Imprime estadísticas comparativas."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS COMPARATIVAS")
    print("="*60)
    
    print(f"\n{'Métrica':<25} {'ZINC250k':<20} {'Generadas':<20}")
    print("-"*65)
    
    # Número de moléculas válidas
    print(f"{'Moléculas válidas':<25} {len(df_original):<20} {len(df_generated):<20}")
    
    # LogP
    print(f"\n--- LogP ---")
    print(f"{'Media':<25} {df_original['LogP'].mean():<20.3f} {df_generated['LogP'].mean():<20.3f}")
    print(f"{'Desv. Estándar':<25} {df_original['LogP'].std():<20.3f} {df_generated['LogP'].std():<20.3f}")
    print(f"{'Mínimo':<25} {df_original['LogP'].min():<20.3f} {df_generated['LogP'].min():<20.3f}")
    print(f"{'Máximo':<25} {df_original['LogP'].max():<20.3f} {df_generated['LogP'].max():<20.3f}")
    
    # MW
    print(f"\n--- Peso Molecular (Da) ---")
    print(f"{'Media':<25} {df_original['MW'].mean():<20.3f} {df_generated['MW'].mean():<20.3f}")
    print(f"{'Desv. Estándar':<25} {df_original['MW'].std():<20.3f} {df_generated['MW'].std():<20.3f}")
    print(f"{'Mínimo':<25} {df_original['MW'].min():<20.3f} {df_generated['MW'].min():<20.3f}")
    print(f"{'Máximo':<25} {df_original['MW'].max():<20.3f} {df_generated['MW'].max():<20.3f}")
    
    print("\n" + "="*60)


def main():
    print("="*60)
    print("ANÁLISIS COMPARATIVO: Moléculas Generadas vs ZINC250k")
    print("="*60)
    
    # 1. Cargar modelo VAE
    print("\n[1/5] Cargando modelo VAE...")
    model, stoi, itos, latent_dim = load_model_and_vocab()
    print(f"  Modelo cargado correctamente (latent_dim={latent_dim})")
    
    # 2. Generar moléculas
    print(f"\n[2/5] Generando moléculas desde el espacio latente...")
    generated_smiles = generate_molecules(model, stoi, itos, latent_dim, NUM_GENERATED)
    
    # 3. Cargar dataset original (muestra del mismo tamaño para comparación justa)
    print(f"\n[3/5] Cargando dataset ZINC250k...")
    original_smiles = load_zinc_dataset(ZINC_PATH, sample_size=NUM_GENERATED)
    
    # 4. Calcular propiedades
    print(f"\n[4/5] Calculando propiedades moleculares...")
    print("  Procesando moléculas originales...")
    df_original = calculate_properties(original_smiles, source_name='ZINC250k (Original)')
    print(f"    -> {len(df_original)} moléculas válidas")
    
    print("  Procesando moléculas generadas...")
    df_generated = calculate_properties(generated_smiles, source_name='Generadas (VAE)')
    print(f"    -> {len(df_generated)} moléculas válidas")
    
    # Calcular tasa de validez
    validity_rate = len(df_generated) / NUM_GENERATED * 100
    print(f"\n  Tasa de validez (generadas): {validity_rate:.1f}%")
    
    # 5. Generar visualización
    print(f"\n[5/5] Generando gráficos comparativos...")
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plot_comparison(df_original, df_generated, OUTPUT_PLOT)
    
    # Imprimir estadísticas
    print_statistics(df_original, df_generated)
    
    print("\n¡Análisis completado!")


if __name__ == "__main__":
    main()
