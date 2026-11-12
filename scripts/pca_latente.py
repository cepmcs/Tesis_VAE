"""
PCA del espacio latente (μ) de moléculas de MOSES (test), sin colorear, en formato PDF.
Ejecutar con un entorno que tenga torch y matplotlib:
    python scripts/pca_latente.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Directorio raíz del proyecto (un nivel arriba de scripts/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from vae_model import MolecularVAE
from vae_model_lstm import MolecularVAE_LSTM

# --- CONFIGURACIÓN ---
MODEL_PATH = os.path.join(ROOT_DIR, "modelos_entrenados", "fase2", "SMILES_LSTM_2_256_300_lr1e4_b64.pth")
MODEL_CLASS = MolecularVAE_LSTM   # MolecularVAE para los modelos GRU
DATA_PATH = os.path.join(ROOT_DIR, "data", "mosesSMILES_processed.pt")
OUTPUT_PLOT = os.path.join(ROOT_DIR, "resultados", "modelo_final", "figuras", "pca_latente.pdf")
NUM_MOLECULES = 30000   # moléculas del test de MOSES
BATCH_SIZE = 500
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paleta de la tesina (misma que plot_distribuciones.py)
COLOR_VAE = "#0072B2"
COLOR_TEXT = "#1A202C"
COLOR_MUTED = "#4A5568"
COLOR_GRID = "#E2E8F0"


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    hyper = checkpoint['hyperparams']
    model = MODEL_CLASS(
        vocab_size=hyper['vocab_size'],
        embed_size=hyper['embed'],
        hidden_size=hyper['hidden'],
        latent_size=hyper['latent'],
        num_layers=hyper.get('num_layers', 1)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def collect_latents(model):
    """Codifica moléculas del test de MOSES y retorna sus μ [N, latent]."""
    test_data = torch.load(DATA_PATH, map_location="cpu")["test_data"]
    torch.manual_seed(SEED)
    test_data = test_data[torch.randperm(len(test_data))[:NUM_MOLECULES]]

    latents = []
    with torch.no_grad():
        for i in range(0, len(test_data), BATCH_SIZE):
            x = test_data[i:i + BATCH_SIZE].to(DEVICE)
            _, mu, _ = model(x)
            latents.append(mu.cpu())
    return torch.cat(latents, dim=0).numpy()


def pca_2d(x):
    """PCA vía SVD. Retorna coordenadas [N, 2] y varianza explicada de PC1 y PC2."""
    x_centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = x_centered @ vt[:2].T
    ratio = (s[:2] ** 2) / (s ** 2).sum()
    return coords, ratio


def plot_pca(coords, ratio, output_path):
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": COLOR_MUTED, "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED, "text.color": COLOR_TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    # rasterized: los puntos van como imagen (dpi de savefig) y el resto (ejes, texto) queda vectorial
    ax.scatter(coords[:, 0], coords[:, 1], s=2, color=COLOR_VAE, alpha=0.35, linewidths=0, rasterized=True)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Gráfico guardado en: {output_path}")


def main():
    print(f"Usando dispositivo: {DEVICE}")
    model = load_model()
    latents = collect_latents(model)
    print(f"Moléculas codificadas: {len(latents)} | dimensiones latentes: {latents.shape[1]}")

    coords, ratio = pca_2d(latents)
    print(f"Varianza explicada: PC1 {ratio[0] * 100:.1f}% | PC2 {ratio[1] * 100:.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plot_pca(coords, ratio, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
