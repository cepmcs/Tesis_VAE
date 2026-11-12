"""
Comparación de distribuciones de QED y SA: moléculas generadas por el VAE vs MOSES (test).
Ejecutar con un entorno que tenga torch, rdkit, seaborn y scipy:
    python scripts/plot_distribuciones.py
"""
import os
import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig
from scipy.stats import wasserstein_distance

# Agregar la ruta de contribuciones de RDKit para SA_Score
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # pyright: ignore[reportMissingImports]

# Directorio raíz del proyecto (un nivel arriba de scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from vae_model_lstm import MolecularVAE_LSTM

# Desactivar warnings de RDKit
RDLogger.DisableLog('rdApp.*')

# --- CONFIGURACIÓN ---
MODEL_NAME = "SMILES_LSTM_2_256_300_lr1e4_b64"
GENERATED_PATH = os.path.join(ROOT_DIR, "resultados", "fase2", "generados", f"{MODEL_NAME}_generated.txt")
MODEL_PATH = os.path.join(ROOT_DIR, "modelos_entrenados", "fase2", f"{MODEL_NAME}.pth")
MOSES_PATH = os.path.join(ROOT_DIR, "data", "moses.csv")
OUTPUT_PLOT = os.path.join(ROOT_DIR, "resultados", "modelo_final", "figuras", "distribuciones_qed_sa.pdf")
NUM_GENERATED = 100_000    # Moléculas generadas (se completan si el archivo tiene menos)
MAX_LEN = 100
TEMP = 1.0
BATCH_SIZE = 5000
N_JOBS = 12
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paleta de la tesina (figBlue, figOrange, darkText, grayText, cardBorder)
COLOR_VAE = "#0072B2"
COLOR_MOSES = "#E69F00"
COLOR_TEXT = "#1A202C"
COLOR_MUTED = "#4A5568"
COLOR_GRID = "#E2E8F0"

SPECIAL_TOKENS = {'[PAD]', '[SOS]', '[EOS]', '[UNK]'}
PLOT_CONFIG = (("QED", "QED"), ("SA", "SA score"))


def load_model_and_vocab():
    """Carga el modelo VAE-LSTM entrenado y los vocabularios."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    hyper = checkpoint['hyperparams']

    model = MolecularVAE_LSTM(
        vocab_size=hyper['vocab_size'],
        embed_size=hyper['embed'],
        hidden_size=hyper['hidden'],
        latent_size=hyper['latent'],
        num_layers=hyper['num_layers']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model, checkpoint['vocab_stoi'], checkpoint['vocab_itos'], hyper['latent']


def generate_smiles(model, vocab_stoi, vocab_itos, latent_dim, num_molecules):
    """Genera SMILES muestreando z ~ N(0, I) y decodificando con muestreo a temperatura TEMP."""
    sos_idx = vocab_stoi['[SOS]']
    eos_idx = vocab_stoi['[EOS]']
    all_smiles = []

    with torch.no_grad():
        for start in range(0, num_molecules, BATCH_SIZE):
            batch = min(BATCH_SIZE, num_molecules - start)
            z = torch.randn(batch, latent_dim, device=DEVICE)
            h = model.decoder_input_h(z).unsqueeze(0).repeat(model.num_layers, 1, 1)
            c = model.decoder_input_c(z).unsqueeze(0).repeat(model.num_layers, 1, 1)

            current_token = torch.full((batch, 1), sos_idx, dtype=torch.long, device=DEVICE)
            finished = torch.zeros(batch, dtype=torch.bool, device=DEVICE)
            decoded = []

            for _ in range(MAX_LEN):
                out, (h, c) = model.decoder_rnn(model.embedding(current_token), (h, c))
                probs = F.softmax(model.fc_out(out.squeeze(1)) / TEMP, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_token = torch.where(finished.unsqueeze(1), torch.zeros_like(next_token), next_token)

                decoded.append(next_token)
                current_token = next_token
                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break

            for seq in torch.cat(decoded, dim=1).cpu().numpy():
                tokens = []
                for idx in seq:
                    if idx == 0:
                        continue
                    token = vocab_itos[int(idx)]
                    if token == '[EOS]':
                        break
                    if token not in SPECIAL_TOKENS:
                        tokens.append(token)
                all_smiles.append("".join(tokens))

    return all_smiles


def load_generated():
    """Lee las moléculas ya generadas y completa hasta NUM_GENERATED si hace falta."""
    smiles = []
    if os.path.exists(GENERATED_PATH):
        with open(GENERATED_PATH, "r", encoding="utf-8") as f:
            smiles = [line.strip() for line in f if line.strip()][:NUM_GENERATED]
    print(f"Moléculas leídas de archivo: {len(smiles)}")

    missing = NUM_GENERATED - len(smiles)
    if missing > 0:
        print(f"Generando {missing} moléculas adicionales...")
        torch.manual_seed(SEED)
        model, stoi, itos, latent_dim = load_model_and_vocab()
        smiles += generate_smiles(model, stoi, itos, latent_dim, missing)

    return smiles


def load_moses_test(path):
    """Carga los SMILES del split de test de MOSES."""
    df = pd.read_csv(path)
    df_test = df[df['SPLIT'] == 'test']
    return df_test['SMILES'].str.strip().tolist()


def _properties(smiles):
    """QED y SA de un SMILES, o None si es inválido."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol), sascorer.calculateScore(mol)


def calculate_properties(smiles_list):
    """Calcula QED y SA de las moléculas válidas. Retorna un DataFrame."""
    with Pool(N_JOBS) as pool:
        records = [r for r in pool.map(_properties, smiles_list, chunksize=500) if r is not None]
    return pd.DataFrame(records, columns=["QED", "SA"])


def plot_comparison(df_moses, df_vae, output_path):
    """Grafica la distribución KDE de QED y SA para MOSES y el VAE."""
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED, "text.color": COLOR_TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })

    fig, axes = plt.subplots(1, len(PLOT_CONFIG), figsize=(8, 3.4))
    for ax, (prop, xlabel) in zip(axes, PLOT_CONFIG):
        sns.kdeplot(df_moses[prop], ax=ax, color=COLOR_MOSES, fill=True, alpha=0.35, linewidth=2, label="MOSES")
        sns.kdeplot(df_vae[prop], ax=ax, color=COLOR_VAE, fill=True, alpha=0.30, linewidth=2, label="VAE")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Densidad" if prop == PLOT_CONFIG[0][0] else "")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Gráfico guardado en: {output_path}")


def main():
    print("[1/4] Cargando moléculas generadas...")
    generated = load_generated()

    print("[2/4] Cargando MOSES (test)...")
    moses = load_moses_test(MOSES_PATH)
    print(f"  MOSES test: {len(moses)} moléculas")

    print("[3/4] Calculando QED y SA...")
    df_vae = calculate_properties(generated)
    df_moses = calculate_properties(moses)
    print(f"  Válidas (VAE): {len(df_vae)} / {len(generated)} = {len(df_vae) / len(generated):.4f}")

    for prop, _ in PLOT_CONFIG:
        print(f"  Wasserstein {prop}: {wasserstein_distance(df_vae[prop], df_moses[prop]):.4f}")

    print("[4/4] Generando gráfico...")
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plot_comparison(df_moses, df_vae, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
