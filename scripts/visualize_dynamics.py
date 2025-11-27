import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
# Mettez ici le chemin correct vers votre dossier de données (si différent)
ROOT_DIR = os.path.join('DigiLeTs', 'data', 'preprocessed', 'complete')

# Choisissez un fichier spécifique à visualiser 
PARTICIPANT_FILE = '007-f-24-right_2019-06-19-12-18-49_preprocessed' 

# Index du caractère à analyser (0-9 pour chiffres, 10-35 pour a-z)
# 10 correspond à la lettre 'a'
CHAR_INDEX = 10 

def visualize_mean_dynamics(file_path, char_index):
    print(f"--> Analyse du fichier : {os.path.basename(file_path)}")
    
    # 1. Lecture du fichier texte
    if not os.path.exists(file_path):
        print(f"ERREUR : Le fichier n'existe pas : {file_path}")
        return

    instances = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        traj_count = 0
        # On cible les 5 instances du caractère demandé (ex: indices 50, 51, 52, 53, 54)
        start_idx = char_index * 5
        end_idx = start_idx + 5
        
        for line in lines:
            parts = line.strip().split()
            # Filtres de sécurité (ignorer labels et lignes vides)
            if len(parts) == 62 or len(parts) % 5 != 0 or len(parts) == 0: 
                continue

            # Si on est dans la plage des instances du caractère cible
            if start_idx <= traj_count < end_idx:
                values = np.array([float(x) for x in parts]).reshape(-1, 5)
                instances.append(values)
            
            traj_count += 1
            # Optimisation : on arrête de lire dès qu'on a dépassé notre caractère
            if traj_count >= end_idx: 
                break
                
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return

    if not instances:
        print(f"Aucune donnée trouvée pour le caractère index {char_index}.")
        print("Vérifiez que le fichier n'est pas corrompu ou vide.")
        return

    print(f"--> {len(instances)} instances trouvées. Calcul des dynamiques...")

    # 2. Calcul des vitesses et Interpolation
    # On normalise toutes les trajectoires à 100 points pour pouvoir faire une moyenne
    NUM_POINTS = 100
    
    vel_mag_list = [] # Liste des courbes de vitesse
    pressure_list = [] # Liste des courbes de pression
    
    for inst in instances:
        # --- Calcul de la Vitesse (Dérivée) ---
        # Colonne 0 = X, Colonne 1 = Y
        vx = np.gradient(inst[:, 0])
        vy = np.gradient(inst[:, 1])
        # Magnitude de la vitesse = racine(vx² + vy²)
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Colonne 2 = Pression
        pressure = inst[:, 2]
        
        # --- Interpolation (Redimensionner à 100 points) ---
        # Crée une échelle de temps de 0 à 1 pour la longueur actuelle
        orig_t = np.linspace(0, 1, len(v_mag))
        # Crée une échelle de temps de 0 à 1 avec 100 points fixes
        target_t = np.linspace(0, 1, NUM_POINTS)
        
        # Interpole les valeurs pour qu'elles collent à l'échelle fixe
        v_interp = np.interp(target_t, orig_t, v_mag)
        p_interp = np.interp(target_t, orig_t, pressure)
        
        vel_mag_list.append(v_interp)
        pressure_list.append(p_interp)

    # 3. Calcul de la Moyenne et de l'Écart-type
    mean_vel = np.mean(vel_mag_list, axis=0)
    std_vel = np.std(vel_mag_list, axis=0)
    
    mean_pres = np.mean(pressure_list, axis=0)
    std_pres = np.std(pressure_list, axis=0)

    # 4. Visualisation
    # Définition explicite de l'axe temporel t
    t = np.linspace(0, 100, NUM_POINTS)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Graphique 1 : Vitesse
    ax1.plot(t, mean_vel, 'b-', linewidth=2, label='Vitesse Moyenne')
    # Zone d'ombre pour la variabilité (écart-type)
    ax1.fill_between(t, mean_vel - std_vel, mean_vel + std_vel, color='b', alpha=0.2, label='Variabilité (Std)')
    # Tracer les essais individuels en transparence
    for v in vel_mag_list:
        ax1.plot(t, v, 'b--', alpha=0.3, linewidth=0.5)
        
    ax1.set_ylabel('Vitesse (Magnitude)')
    ax1.set_title(f"Dynamique de Vitesse - Participant {os.path.basename(file_path)[:3]} - Caractère Index {char_index}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graphique 2 : Pression
    ax2.plot(t, mean_pres, 'r-', linewidth=2, label='Pression Moyenne')
    ax2.fill_between(t, mean_pres - std_pres, mean_pres + std_pres, color='r', alpha=0.2)
    for p in pressure_list:
        ax2.plot(t, p, 'r--', alpha=0.3, linewidth=0.5)
        
    ax2.set_ylabel('Pression')
    ax2.set_xlabel('Temps Normalisé (%)')
    ax2.set_title(f"Dynamique de Pression")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    output_filename = f"dynamics_char{char_index}.png"
    plt.savefig(output_filename)
    print(f" Graphique généré avec succès : {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Construction du chemin complet
    # Si le script est lancé depuis DijiLets, ROOT_DIR pointe vers data/...
    full_path = os.path.join(ROOT_DIR, PARTICIPANT_FILE)
    
    # Vérification alternative si on ne trouve pas le fichier
    if not os.path.exists(full_path):
        # Essayer de trouver le fichier dans le dossier courant pour tester
        if os.path.exists(PARTICIPANT_FILE):
            full_path = PARTICIPANT_FILE
        else:
            # Essayer le chemin absolu si besoin
            print(f" Fichier introuvable dans : {full_path}")
            print("Tentative de recherche manuelle...")
            # Exemple de chemin 'hardcodé' pour votre machine si le relatif échoue
            full_path = os.path.join(r'C:\Users\User\OneDrive\Desktop\DijiLets\DigiLeTs\data\preprocessed\complete', PARTICIPANT_FILE)

    visualize_mean_dynamics(full_path, CHAR_INDEX)