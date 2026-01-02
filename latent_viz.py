import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from vae_model import MolecularVAE

# Configuración por defecto (sin CLI)
MODEL_PATH = "vae_model.pth"
DATA_PATH = "data_processed.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
MAX_SAMPLES = 5000  # usa todas si es menor que el dataset
OUT_PNG = "latent_pca.png"
SAVE_NPY = None  # opcional: "latent_coords.npy"
SEED = 0


def load_model(model_path: str, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    hyper = ckpt.get("hyperparams", {})
    model = MolecularVAE(
        vocab_size=hyper.get("vocab_size", len(ckpt["vocab_stoi"])),
        embed_size=hyper.get("embed", 128),
        hidden_size=hyper.get("hidden", 128),
        latent_size=hyper.get("latent", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_data(data_path: str):
    saved = torch.load(data_path, map_location="cpu")
    return saved["data"]


def collect_latents(model, data_tensor, batch_size, device):
    loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)
    latents, lengths = [], []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _, mu, _ = model(x)
            latents.append(mu.cpu())
            lengths.append((x != 0).sum(dim=1).cpu())
    latents = torch.cat(latents, dim=0).numpy()
    lengths = torch.cat(lengths, dim=0).numpy()
    return latents, lengths


def pca_2d(x: np.ndarray):
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = x_centered @ vt[:2].T
    var = (s ** 2) / (x.shape[0] - 1)
    ratio = var[:2] / var.sum()
    return coords, ratio


def plot(coords, lengths, ratio, out_path):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], s=8, c=lengths, cmap="viridis", alpha=0.6)
    plt.colorbar(sc, label="Longitud de secuencia")
    plt.xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    plt.title("Espacio latente (mu) - PCA")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Gráfico guardado en {out_path}")


def main():
    print(f"Usando dispositivo: {DEVICE}")
    data_tensor = load_data(DATA_PATH)

    if MAX_SAMPLES and MAX_SAMPLES < len(data_tensor):
        torch.manual_seed(SEED)
        idx = torch.randperm(len(data_tensor))[:MAX_SAMPLES]
        data_tensor = data_tensor[idx]
        print(f"Muestras usadas: {len(data_tensor)} (subsample)")
    else:
        print(f"Muestras usadas: {len(data_tensor)}")

    model = load_model(MODEL_PATH, DEVICE)
    latents, lengths = collect_latents(model, data_tensor, BATCH_SIZE, DEVICE)
    coords, ratio = pca_2d(latents)
    plot(coords, lengths, ratio, OUT_PNG)

    if SAVE_NPY:
        np.save(SAVE_NPY, {"coords": coords, "lengths": lengths})
        print(f"Coords guardadas en {SAVE_NPY}")


if __name__ == "__main__":
    main()
