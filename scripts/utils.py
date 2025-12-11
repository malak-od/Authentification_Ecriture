import os
import numpy as np

def read_digilets_file(filepath):
    """
    Lit un fichier DigiLeTs de manière robuste (insensible aux erreurs de format).
    Retourne une matrice NumPy (N_points, 5).
    """
    all_values = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            # On ignore les lignes de labels (62 colonnes) ou vides
            if len(parts) == 62 or len(parts) == 0:
                continue
            try:
                # On ajoute les chiffres à la suite dans une liste plate
                all_values.extend([float(x) for x in parts])
            except ValueError:
                continue

        # Reformatage en matrice
        raw_array = np.array(all_values)
        
        # Sécurité : on s'assure que le total est un multiple de 5
        remainder = len(raw_array) % 5
        if remainder != 0:
            raw_array = raw_array[:-remainder]
            
        if len(raw_array) == 0: return None
        
        # 5 colonnes : X, Y, Pression, PenDown, Time
        return raw_array.reshape(-1, 5)

    except Exception as e:
        print(f"Erreur utils.read_digilets_file: {e}")
        return None

def compute_dynamics(trajectory):
    """
    Calcule la physique du mouvement (Vitesse, Accélération).
    Entrée : (N, 5) -> Sortie : (N, 9) normalisée entre 0 et 1
    """
    # 1. Calcul des dérivées (Vitesse)
    vx = np.gradient(trajectory[:, 0])
    vy = np.gradient(trajectory[:, 1])
    vmag = np.sqrt(vx**2 + vy**2) # Magnitude
    
    # 2. Calcul des dérivées secondes (Accélération)
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    
    # 3. Fusion des caractéristiques [X, Y, P, Pen, Vx, Vy, Vmag, Ax, Ay]
    features = np.column_stack((trajectory[:, :4], vx, vy, vmag, ax, ay))
    
    # 4. Normalisation (0 à 1) indispensable pour le Rate Coding
    denom = features.max(axis=0) - features.min(axis=0)
    denom[denom == 0] = 1.0 # Évite la division par zéro
    
    features_norm = (features - features.min(axis=0)) / denom
    
    return features_norm