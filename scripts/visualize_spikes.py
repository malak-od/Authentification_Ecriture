import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# On importe nos outils habituels
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_digilets_file, compute_dynamics

# === CONFIGURATION ===
# Mettez ici le chemin vers un fichier du participant 002 (ou n'importe quel autre)
# C'est le même fichier que pour visualize_dynamics.py
FILE_PATH = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete\002-f-22-right_2019-06-05-12-21-29_preprocessed"

def rate_coding(data):
    """
    Simule l'encodage du SNN : 
    Transforme les valeurs continues (0.8) en Spikes (0 ou 1) via probabilité.
    """
    # Convertit numpy -> tensor pytorch
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    # Génère les spikes (Bernoulli)
    return (data > torch.rand_like(data)).float()

def visualize_spikes():
    print(f"Analyse du fichier : {os.path.basename(FILE_PATH)}")
    
    # 1. Lecture
    if not os.path.exists(FILE_PATH):
        print(f" Fichier introuvable : {FILE_PATH}")
        return

    raw_data = read_digilets_file(FILE_PATH)
    if raw_data is None: return

    # 2. On isole UN SEUL exemple (le premier chiffre "0")
    # Un chiffre dure environ 40 à 60 points. Prenons les 50 premiers points.
    sample_length = 1000
    sample_raw = raw_data[sample_length:sample_length+50]

    # 3. Calcul de la Physique (Features)
    # [X, Y, P, Pen, Vx, Vy, Vmag, Ax, Ay] -> 9 caractéristiques
    features = compute_dynamics(sample_raw)
    
    # 4. Génération des Spikes
    spikes = rate_coding(features)
    
    # --- VISUALISATION ---
    features_np = features # (50, 9)
    spikes_np = spikes.numpy()   # (50, 9)
    
    feature_names = ['X', 'Y', 'Pression', 'PenDown', 'Vit X', 'Vit Y', 'Vit Mag', 'Acc X', 'Acc Y']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Graphique 1 : Les Valeurs Continues (Heatmap)
    # On transpose (.T) pour avoir le Temps en axe X et les Features en axe Y
    im1 = ax1.imshow(features_np.T, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax1.set_title("1. Ce que voit l'Humain : Données Physiques Normalisées (0.0 à 1.0)")
    ax1.set_yticks(range(9))
    ax1.set_yticklabels(feature_names)
    plt.colorbar(im1, ax=ax1, label="Intensité")

    # Graphique 2 : Les Spikes (Raster Plot)
    # Noir = Pas de Spike, Blanc = Spike
    im2 = ax2.imshow(spikes_np.T, aspect='auto', cmap='gray', origin='lower', interpolation='nearest')
    ax2.set_title("2. Ce que voit le SNN : Trains de Spikes (0 ou 1)")
    ax2.set_yticks(range(9))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel("Temps (points)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_spikes()