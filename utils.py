import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_loss(train_losses, val_losses, val_interval, save_path):
    
    epochs = list(range(1, len(train_losses) + 1))
    val_epochs = list(range(val_interval, len(train_losses) + 1, val_interval))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(val_epochs, val_losses, label='Val Loss', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

