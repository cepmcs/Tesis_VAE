import os
import sys
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import selfies as sf
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, RDConfig

# Agregar la ruta de contribuciones de RDKit para SA_Score
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # pyright: ignore[reportMissingImports]

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar arquitectura del VAE
from vae_model import MolecularVAE

# Desactivar warnings de RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "vae_model.pth")
MOSES_PATH = os.path.join(ROOT_DIR, "data", "moses.csv")
NUM_GENERATED = 30000      # Número de moléculas a generar
MAX_LEN = 100
TEMP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PLOT = os.path.join(ROOT_DIR, "outputs", "comparacion_propiedades.png")
SOURCE_ORIGINAL = "MOSES (Original)"
SOURCE_GENERATED = "Generadas (VAE)"
PLOT_COLORS = {SOURCE_ORIGINAL: "#2E86AB", SOURCE_GENERATED: "#E94F37"}

PLOT_CONFIG: Sequence[Tuple[str, str, str]] = (
    ("LogP", "LogP", "Distribución de LogP"),
    ("MW", "Peso Molecular (Da)", "Distribución de Peso Molecular"),
    ("QED", "QED", "Distribución de QED"),
    ("SA", "SA Score", "Accesibilidad Sintética (SA)"),
)

STATS_CONFIG: Dict[str, Sequence[Tuple[str, str]]] = {
    "LogP": (("Media", "mean"), ("Desv. Estándar", "std"), ("Mínimo", "min"), ("Máximo", "max")),
    "MW": (("Media", "mean"), ("Desv. Estándar", "std"), ("Mínimo", "min"), ("Máximo", "max")),
    "QED": (("Media", "mean"), ("Desv. Estándar", "std")),
    "SA": (("Media", "mean"), ("Desv. Estándar", "std")),
}


def load_model_and_vocab():
    """Carga el modelo VAE entrenado y los vocabularios."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

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


def decode_latent(model, z, vocab_stoi, max_len=100, temp=1.0):
    """Decodifica vectores latentes a secuencias de índices."""
    if temp <= 0:
        raise ValueError("La temperatura de decodificación debe ser mayor a 0.")

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
        except Exception:
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
        indices = decode_latent(model, z, vocab_stoi, MAX_LEN, TEMP)
        smiles = indices_to_smiles(indices, vocab_itos)
        all_smiles.extend(smiles)
        
        if (i + 1) % 5 == 0 or i == num_batches - 1:
            print(f"  Procesados {min((i+1) * batch_size, num_molecules)} / {num_molecules}")
    
    return all_smiles


def calculate_properties(smiles_list, source_name=""):
    """
    Calcula Peso Molecular (MW), LogP, QED y SA para una lista de SMILES.
    Retorna un DataFrame con las propiedades.
    """
    records = []

    for sm in smiles_list:
        if sm is None:
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            try:
                records.append(
                    {
                        'SMILES': sm,
                        'MW': Descriptors.MolWt(mol),
                        'LogP': Descriptors.MolLogP(mol),
                        'QED': QED.qed(mol),
                        'SA': sascorer.calculateScore(mol),
                        'Source': source_name,
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(records)


def load_moses_dataset(path, sample_size=None):
    """Carga el dataset MOSES y extrae SMILES."""
    print(f"Cargando dataset MOSES desde {path}...")
    df = pd.read_csv(path)

    required_cols = {'SMILES', 'SPLIT'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f"Faltan columnas requeridas en MOSES: {missing}")
    
    # MOSES usa la columna "SMILES" y "SPLIT". Filtrar por train
    df_train = df[df['SPLIT'] == 'train'].copy()
    
    # Limpiar SMILES (quitar saltos de línea)
    df_train['SMILES'] = df_train['SMILES'].str.strip()
    
    if sample_size and sample_size < len(df_train):
        df_train = df_train.sample(n=sample_size, random_state=42)
        print(f"  Muestra de {sample_size} moléculas seleccionada")
    
    return df_train['SMILES'].tolist()


def _plot_property_distribution(ax, df_combined, df_original, df_generated, prop, xlabel, title):
    """Grafica la distribución KDE de una propiedad para ambos conjuntos."""
    for source in (SOURCE_ORIGINAL, SOURCE_GENERATED):
        data = df_combined[df_combined['Source'] == source][prop]
        sns.kdeplot(
            data=data,
            ax=ax,
            label=source,
            color=PLOT_COLORS[source],
            linewidth=2,
            fill=True,
            alpha=0.3,
        )

    ax.axvline(
        x=df_original[prop].mean(),
        color=PLOT_COLORS[SOURCE_ORIGINAL],
        linestyle='--',
        alpha=0.7,
    )
    ax.axvline(
        x=df_generated[prop].mean(),
        color=PLOT_COLORS[SOURCE_GENERATED],
        linestyle='--',
        alpha=0.7,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)


def plot_comparison(df_original, df_generated, output_path):
    """Genera gráficos KDE comparativos de MW, LogP, QED y SA."""

    # Combinar dataframes
    df_combined = pd.concat([df_original, df_generated], ignore_index=True)

    # Configurar estilo
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (prop, xlabel, title) in zip(axes.flatten(), PLOT_CONFIG):
        _plot_property_distribution(
            ax=ax,
            df_combined=df_combined,
            df_original=df_original,
            df_generated=df_generated,
            prop=prop,
            xlabel=xlabel,
            title=title,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nGráfico guardado en: {output_path}")


def _print_metric_statistics(df_original, df_generated, metric_name, title):
    """Imprime estadísticas configuradas para una métrica."""
    print(f"\n--- {title} ---")
    for label, method_name in STATS_CONFIG[metric_name]:
        original_value = getattr(df_original[metric_name], method_name)()
        generated_value = getattr(df_generated[metric_name], method_name)()
        print(f"{label:<25} {original_value:<20.3f} {generated_value:<20.3f}")


def print_statistics(df_original, df_generated):
    """Imprime estadísticas comparativas."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS COMPARATIVAS")
    print("="*60)
    
    print(f"\n{'Métrica':<25} {'MOSES':<20} {'Generadas':<20}")
    print("-"*65)

    # Número de moléculas válidas
    print(f"{'Moléculas válidas':<25} {len(df_original):<20} {len(df_generated):<20}")

    metric_titles = {
        'LogP': 'LogP',
        'MW': 'Peso Molecular (Da)',
        'QED': 'QED (Drug-likeness)',
        'SA': 'SA Score (Synthetic Accessibility)',
    }
    for metric_name, title in metric_titles.items():
        _print_metric_statistics(df_original, df_generated, metric_name, title)

    print("\n" + "="*60)


def main():
    print("="*60)
    print("ANÁLISIS COMPARATIVO: Moléculas Generadas vs MOSES")
    print("="*60)
    
    # 1. Cargar modelo VAE
    print("\n[1/5] Cargando modelo VAE...")
    model, stoi, itos, latent_dim = load_model_and_vocab()
    print(f"  Modelo cargado correctamente (latent_dim={latent_dim})")
    
    # 2. Generar moléculas
    print(f"\n[2/5] Generando moléculas desde el espacio latente...")
    generated_smiles = generate_molecules(model, stoi, itos, latent_dim, NUM_GENERATED)
    
    # 3. Cargar dataset original (muestra del mismo tamaño para comparación justa)
    print(f"\n[3/5] Cargando dataset MOSES...")
    original_smiles = load_moses_dataset(MOSES_PATH, sample_size=NUM_GENERATED)
    
    # 4. Calcular propiedades
    print(f"\n[4/5] Calculando propiedades moleculares...")
    print("  Procesando moléculas originales...")
    df_original = calculate_properties(original_smiles, source_name=SOURCE_ORIGINAL)
    print(f"    -> {len(df_original)} moléculas válidas")

    print("  Procesando moléculas generadas...")
    df_generated = calculate_properties(generated_smiles, source_name=SOURCE_GENERATED)
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
