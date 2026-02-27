import matplotlib.pyplot as plt

def plot_training(history, save_path="training_loss.png"):
    """
    Genera gráficos de loss y accuracy del entrenamiento.
    
    Args:
        history: Diccionario con las métricas de entrenamiento
        save_path: Ruta donde guardar la imagen
    """
    print("Generando gráfico de entrenamiento...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss por token')
    ax1.plot(history['train_loss'], label='Train Loss', color='tab:red')
    ax1.plot(history['val_loss'], label='Val Loss', color='tab:orange', linestyle='--')
    ax1.set_title('Loss (Train vs Validación)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Accuracy
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.plot(history['train_accuracy'], label='Train Acc', color='tab:blue')
    ax2.plot(history['val_accuracy'], label='Val Acc', color='tab:cyan', linestyle='--')
    ax2.set_title('Accuracy (Train vs Validación)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Gráfica guardada en {save_path}")
