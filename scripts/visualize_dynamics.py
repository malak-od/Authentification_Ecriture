import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Importation de la fonction de calcul physique depuis utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import compute_dynamics

# === CONFIGURATION ===
# Mettez votre chemin absolu ici aussi
ROOT_DIR = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete"

# Fichier et Caractère à visualiser
PARTICIPANT_FILE = '002-f-22-right_2019-06-05-12-21-29_preprocessed' 
CHAR_INDEX = 10  # Index 10 = 'a'

def visualize_mean_dynamics(file_path, char_index):
    print(f"--> Analyse du fichier : {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f" ERREUR : Fichier introuvable : {file_path}")
        return

    # --- 1. LECTURE SPÉCIFIQUE (Pour isoler le caractère) ---
    instances = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        traj_count = 0
        # On vise les 5 essais du caractère demandé
        start_idx = char_index * 5
        end_idx = start_idx + 5
        
        for line in lines:
            parts = line.strip().split()
            # On ignore les labels (62) ou lignes vides
            if len(parts) == 62 or len(parts) == 0: 
                continue
            
            # Robustesse : on ignore les lignes bizarres
            if len(parts) % 5 != 0:
                continue

            # Si on est dans la zone du caractère cible
            if start_idx <= traj_count < end_idx:
                try:
                    vals = np.array([float(x) for x in parts]).reshape(-1, 5)
                    instances.append(vals)
                except ValueError:
                    continue
            
            traj_count += 1
            if traj_count >= end_idx: 
                break
                
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return

    if not instances:
        print(f" Aucune donnée trouvée pour le caractère {char_index}.")
        return

    print(f"--> {len(instances)} instances trouvées. Calcul via utils.py...")

    # --- 2. CALCUL ET INTERPOLATION ---
    NUM_POINTS = 100
    vel_mag_list = []
    pressure_list = []
    
    for raw_inst in instances:
        # APPEL À UTILS : On utilise la même physique que pour l'entraînement !
        # compute_dynamics renvoie : [X, Y, P, Pen, Vx, Vy, Vmag, Ax, Ay]
        # (Attention : compute_dynamics normalise les données, ce qui est très bien)
        feats = compute_dynamics(raw_inst)
        
        # Extraction : Colonne 6 = Vmag, Colonne 2 = Pression (dans raw_inst c'était 2, dans feats c'est 2 aussi)
        # Rappel structure compute_dynamics : trajectory[:, :4] (0,1,2,3) + vx, vy, vmag, ax, ay
        # Donc Pression = index 2
        # Vmag = index 6
        
        v_mag = feats[:, 6]
        pressure = feats[:, 2] # Pression (normalisée par utils)
        
        # Interpolation pour le graphique
        orig_t = np.linspace(0, 1, len(v_mag))
        target_t = np.linspace(0, 1, NUM_POINTS)
        
        vel_mag_list.append(np.interp(target_t, orig_t, v_mag))
        pressure_list.append(np.interp(target_t, orig_t, pressure))

    # --- 3. MOYENNE ET ÉCART-TYPE ---
    mean_vel = np.mean(vel_mag_list, axis=0)
    std_vel = np.std(vel_mag_list, axis=0)
    
    mean_pres = np.mean(pressure_list, axis=0)
    std_pres = np.std(pressure_list, axis=0)

    # --- 4. VISUALISATION ---
    t = np.linspace(0, 100, NUM_POINTS)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Graphique Vitesse
    ax1.plot(t, mean_vel, 'b-', linewidth=2, label='Vitesse Moyenne')
    ax1.fill_between(t, mean_vel - std_vel, mean_vel + std_vel, color='b', alpha=0.2)
    for v in vel_mag_list:
        ax1.plot(t, v, 'b--', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Vitesse (Normalisée)')
    ax1.set_title(f"Profil Dynamique - Participant {PARTICIPANT_FILE[:3]} - Caractère {char_index}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graphique Pression
    ax2.plot(t, mean_pres, 'r-', linewidth=2, label='Pression Moyenne')
    ax2.fill_between(t, mean_pres - std_pres, mean_pres + std_pres, color='r', alpha=0.2)
    for p in pressure_list:
        ax2.plot(t, p, 'r--', alpha=0.3, linewidth=0.5)
    ax2.set_ylabel('Pression (Normalisée)')
    ax2.set_xlabel('Temps (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    full_path = os.path.join(ROOT_DIR, PARTICIPANT_FILE)
    visualize_mean_dynamics(full_path, CHAR_INDEX)