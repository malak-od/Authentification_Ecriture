import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DigiLeTsDataset(Dataset):
    def __init__(self, root_dir, num_steps=100, mode='train'):
        """
        Args:
            root_dir (str): Chemin vers le dossier 'data/preprocessed/complete'
            num_steps (int): Nombre de pas de temps pour normaliser la longueur (pour les SNN/RNN)
        """
        self.root_dir = root_dir
        self.num_steps = num_steps
        self.files = glob.glob(os.path.join(root_dir, '*_preprocessed'))
        self.files = [f for f in self.files if not f.endswith('_info')]
        
        self.samples = []
        self._load_all_data()

    def _compute_derivatives(self, trajectory):
        """
        Calcule Vitesse et Accélération.
        Entrée: (L, 5) -> [X, Y, Pression, PenDown, Time]
        Sortie: (L, 9) -> [X, Y, Pression, PenDown, Vx, Vy, V_mag, Ax, Ay]
        """
        # Extraction
        X = trajectory[:, 0]
        Y = trajectory[:, 1]
        # Pression et PenDown ne sont pas dérivés
        
        # 1. Vitesse (Dérivée première : delta position)
        # On utilise np.gradient qui gère bien les bords
        Vx = np.gradient(X)
        Vy = np.gradient(Y)
        V_mag = np.sqrt(Vx**2 + Vy**2) # Vitesse scalaire (magnitude)

        # 2. Accélération (Dérivée seconde : delta vitesse)
        Ax = np.gradient(Vx)
        Ay = np.gradient(Vy)

        # Concaténation des nouvelles caractéristiques
        # On garde les 4 premières (X, Y, P, Pen) et on ajoute les dynamiques
        # Shape finale : (L, 9)
        new_features = np.column_stack((trajectory[:, :4], Vx, Vy, V_mag, Ax, Ay))
        return new_features

    def _resample_trajectory(self, trajectory):
        """Interpole la trajectoire pour avoir une longueur fixe (num_steps)."""
        length = trajectory.shape[0]
        features = trajectory.shape[1]
        
        original_steps = np.linspace(0, 1, length)
        target_steps = np.linspace(0, 1, self.num_steps)
        
        resampled = np.zeros((self.num_steps, features))
        for f in range(features):
            resampled[:, f] = np.interp(target_steps, original_steps, trajectory[:, f])
            
        return resampled

    def _load_all_data(self):
        """Lit tous les fichiers et stocke les échantillons."""
        print(f"Chargement de {len(self.files)} participants...")
        
        for filepath in self.files:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Logique de lecture robuste (adaptée de notre script précédent)
                # Chaque trajectoire est une instance d'un caractère
                current_traj = []
                label_idx = 0 # 0 à 61 (Symboles)
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 62: # Ligne de Label (on change de symbole/instance)
                        continue
                    if len(parts) % 5 != 0 or len(parts) == 0:
                        continue

                    # Conversion
                    values = np.array([float(x) for x in parts])
                    points = values.reshape(-1, 5) # (L, 5)
                    
                    # Traitement
                    # 1. Calcul des dérivées
                    points_dynamic = self._compute_derivatives(points)
                    # 2. Normalisation temporelle (très important pour le batching PyTorch)
                    points_resampled = self._resample_trajectory(points_dynamic)
                    
                    # Conversion en Tensor PyTorch (Float)
                    tensor_data = torch.from_numpy(points_resampled).float()
                    
                    # On ajoute à la liste (Data, Label)
                    # Ici on simule un label simple, à améliorer selon votre logique de symbols
                    label = label_idx // 5 # 5 instances par symbole
                    self.samples.append((tensor_data, label))
                    
                    label_idx += 1
                    
            except Exception as e:
                print(f"Erreur sur {os.path.basename(filepath)}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retourne (Sequence, Label)
        return self.samples[idx]

# Exemple d'utilisation rapide
if __name__ == "__main__":
    # Test
    dataset = DigiLeTsDataset(os.path.join('DigiLeTs', 'data', 'preprocessed', 'complete'))
    print(f"Dataset créé avec {len(dataset)} échantillons.")
    x, y = dataset[0]
    print(f"Forme d'une donnée : {x.shape}") # Devrait être (100, 9)