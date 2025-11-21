import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# On cherche les fichiers à partir de l'endroit où le script est lancé
INPUT_DIR = os.path.join('DigiLeTs', 'data', 'preprocessed', 'complete')
OUTPUT_DIR = 'visualization'

# Dimensions des données DigiLeTs
NUM_SYMBOLS = 62
NUM_INSTANCES = 5
MAX_LENGTH = 250

# CORRECTION : Les fichiers texte ne contiennent que 5 colonnes
# (X, Y, Pression, PenDown, Timestamp)
NUM_FEATURES = 5 

def read_text_file(filepath):
    """
    Lit un fichier texte DigiLeTs et extrait les trajectoires.
    Gère automatiquement le saut des lignes de labels (taille 62).
    """
    trajectories = np.zeros((NUM_SYMBOLS, NUM_INSTANCES, MAX_LENGTH, NUM_FEATURES))
    lengths = np.zeros((NUM_SYMBOLS, NUM_INSTANCES), dtype=int)
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        traj_count = 0
        
        for line in lines:
            parts = line.strip().split()
            
            # Filtre de sécurité : ligne vide
            if len(parts) == 0:
                continue
                
            # Conversion en nombres
            try:
                values = np.array([float(x) for x in parts])
            except ValueError:
                continue
            
            # --- FILTRE CRITIQUE ---
            # Si la ligne ne contient pas un multiple exact de 5 éléments,
            # c'est une ligne de Label (62 éléments) ou corrompue. On l'ignore.
            if len(values) % NUM_FEATURES != 0:
                continue

            # Si on a déjà rempli toutes les cases (310 caractères), on arrête
            if traj_count >= NUM_SYMBOLS * NUM_INSTANCES:
                break

            # Reformatage en (N points, 5 features)
            points = values.reshape(-1, NUM_FEATURES)
            
            # Calcul de la position dans la grille
            symbol_idx = traj_count // NUM_INSTANCES
            instance_idx = traj_count % NUM_INSTANCES
            
            # Stockage (tronqué à 250 points max)
            length = min(len(points), MAX_LENGTH)
            trajectories[symbol_idx, instance_idx, :length, :] = points[:length, :]
            lengths[symbol_idx, instance_idx] = length
            
            traj_count += 1
            
        if traj_count == 0:
            print(f"Aucune trajectoire valide trouvée dans {os.path.basename(filepath)}")
            return None

        return {"trajectories": trajectories, "lengths": lengths}

    except Exception as e:
        print(f"Erreur de lecture sur {filepath}: {e}")
        return None

def plot_instance(instance, ax):
    """Trace une seule lettre."""
    ax.set_facecolor("white")
    
    # Extraction des colonnes (0 à 4)
    X, Y = instance[:, 0], instance[:, 1]
    Pressure = instance[:, 2]
    PenDown = instance[:, 3]
    
    # --- TRACE DES LIGNES ---
    for i in range(len(instance) - 1):
        # Jaune si stylet posé, Noir sinon
        color = "yellow" if PenDown[i] == 1 else "black"
            
        ax.plot([X[i], X[i+1]], [Y[i], Y[i+1]],
                linewidth=Pressure[i]*10 + 0.5,
                color=color, alpha=0.6)
        
    # Note: Sans le fichier _info, on ne peut pas afficher les points
    # verts/magentas (extrapolés/nettoyés), mais la forme sera correcte.

    ax.axis('off')
    return ax

def generate_grid(participant_data, output_filename):
    """Génère la grille complète 62x5."""
    # Création d'une grande figure verticale
    fig, axs = plt.subplots(nrows=NUM_SYMBOLS, ncols=NUM_INSTANCES, 
                            figsize=(10, 100)) 
    
    for s in range(NUM_SYMBOLS):
        for i in range(NUM_INSTANCES):
            length = participant_data["lengths"][s, i]
            if length > 0:
                traj = participant_data["trajectories"][s, i, :length]
                plot_instance(traj, axs[s, i])
            else:
                axs[s, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=100)
    plt.close()
    print(f"Image sauvegardée : {output_filename}")

# ================= MAIN =================
def main():
    # Vérification intelligente du dossier
    if not os.path.exists(INPUT_DIR):
        # Si on est déjà DANS le sous-dossier, on tente le chemin relatif court
        INPUT_DIR_ALT = os.path.join('data', 'preprocessed', 'complete')
        if os.path.exists(INPUT_DIR_ALT):
            input_path = INPUT_DIR_ALT
        else:
            print(f"Erreur : Dossier introuvable : {INPUT_DIR}")
            print("Vérifiez que vous êtes à la racine du projet 'DijiLeTs'.")
            return
    else:
        input_path = INPUT_DIR

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # On cherche tous les fichiers (en ignorant les _info)
    files = glob.glob(os.path.join(input_path, '*_preprocessed'))
    files = [f for f in files if not f.endswith('_info')]

    print(f"Traitement de {len(files)} fichiers texte...")

    for idx, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        # Optionnel : afficher pour suivre la progression
        if idx % 5 == 0: 
            print(f"[{idx+1}/{len(files)}] Traitement de {filename}...")
        
        data = read_text_file(filepath)
        
        if data is not None:
            output_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
            generate_grid(data, output_path)

    print("\nTerminé ! Vérifiez le dossier 'visualization'.")

if __name__ == "__main__":
    main()