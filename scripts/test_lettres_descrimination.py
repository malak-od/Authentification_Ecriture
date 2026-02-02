# -*- coding: utf-8 -*-
import os
import glob
import torch
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# --- 1. CONFIGURATION ---
DATA_DIR = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete"

# Fallback si le chemin n'existe pas (pour test local)
if not os.path.exists(DATA_DIR):
    DATA_DIR = "./data/preprocessed/complete"

TARGET_ID = '002' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import des modules locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from snn_engine import AuthSNN, rate_coding
    from utils import compute_dynamics
except ImportError:
    print("[ERREUR] Il manque snn_engine.py ou utils.py dans ce dossier !")
    sys.exit()

# --- 2. LE DATASET CORRIGÉ ET ROBUSTE ---
class BalancedCharDataset(Dataset):
    def __init__(self, root_dir, target_id, char_index, length=20):
        self.samples = []
        target_samples = []
        imposter_samples = []
        
        if not os.path.exists(root_dir): return

        files = glob.glob(os.path.join(root_dir, '*_preprocessed'))
        # On exclut les fichiers info
        files = [f for f in files if not f.endswith('_info')]
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # Vérifie si c'est la cible ou un imposteur
            is_target = filename.startswith(target_id)
            label = 1.0 if is_target else 0.0
            
            try:
                with open(filepath, 'r') as f: lines = f.readlines()
            except: continue

            # On vise les lignes du caractère spécifique (index * 10 lignes brutes)
            # Chaque caractère occupe 10 lignes dans le fichier (5 essais x 2 lignes data/label)
            start = char_index * 10
            end = start + 10
            if len(lines) < end: continue
            
            # LIGNES PAIRES UNIQUEMENT (Les données sont sur les lignes paires)
            data_lines = lines[start:end][0::2]
            
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                try:
                    vals = np.array([float(x) for x in parts])
                    # Le format est [x, y, p, az, alt] répété
                    num_points = len(vals) // 5
                    traj = vals[:num_points*5].reshape(-1, 5)
                    
                    # Calcul dynamique (Vitesse/Accélération) via utils.py
                    feat = compute_dynamics(traj)
                    if len(feat) == 0: continue
                    
                    # Normalisation temporelle (Padding ou Coupe à 'length')
                    final_seq = np.zeros((length, feat.shape[1]))
                    slen = min(len(feat), length)
                    final_seq[:slen, :] = feat[:slen, :]
                    
                    item = (torch.FloatTensor(final_seq), torch.tensor(label).float())
                    
                    if is_target: target_samples.append(item)
                    else: imposter_samples.append(item)
                except: continue

        # --- SUR-ÉCHANTILLONNAGE (OVERSAMPLING) ---
        # Stratégie : On duplique les cibles pour égaler le nombre d'imposteurs
        if len(target_samples) > 0 and len(imposter_samples) > 0:
            count_target = len(target_samples)
            count_imposter = len(imposter_samples)
            
            # Combien de fois dupliquer ?
            factor = count_imposter // count_target
            remainder = count_imposter % count_target
            
            balanced_targets = target_samples * factor + target_samples[:remainder]
            
            # Fusion : 50% Target / 50% Imposteurs
            self.samples = balanced_targets + imposter_samples
            random.shuffle(self.samples) 
        else:
            self.samples = []

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 3. FONCTION D'EVALUATION SCIENTIFIQUE (Train/Test Split) ---
def evaluate_char(char_idx):
    ds = BalancedCharDataset(DATA_DIR, TARGET_ID, char_idx)
    
    # Si pas assez de données, on renvoie 50% (hasard)
    if len(ds) < 10: return 50.0 
    
    # SPLIT 80% Train / 20% Test (CRUCIAL pour la validité)
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Initialisation du modèle (9 inputs car compute_dynamics ajoute v, a, etc.)
    # Si votre compute_dynamics renvoie 5 ou 7, ajustez num_inputs ici.
    model = AuthSNN(num_inputs=9).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    from spikingjelly.activation_based import functional
    
    # --- PHASE D'ENTRAÎNEMENT ---
    model.train()
    epochs = 15 # Un peu plus d'epochs pour bien apprendre
    for _ in range(epochs): 
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            # Encodage Rate Coding + Forward
            out = model(rate_coding(x)).mean(0).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            
    # --- PHASE DE TEST (Sur données jamais vues) ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(rate_coding(x)).mean(0).squeeze(-1)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            functional.reset_net(model)
            
    return 100 * correct / total if total > 0 else 50.0

# --- 4. MAIN ---
# --- 4. MAIN ---
CHAR_LABELS = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]

if __name__ == "__main__":
    print(f"--- ANALYSE DE DISCRIMINABILITÉ (Sujet {TARGET_ID}) ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"[ERREUR] Dossier introuvable : {DATA_DIR}")
        sys.exit()
    else:
        print(f"[OK] Chargement depuis : {DATA_DIR}")

    print("Calcul des scores par caractère (Train/Test Split 80/20)...")
    results = []
    
    # Boucle sur les 62 caractères
    for i in range(62):
        acc = evaluate_char(i)
        results.append(acc)
        # Feedback visuel
        bar = "#" * int(acc // 10)
        print(f"Char '{CHAR_LABELS[i]}' : {acc:.2f}% \t[{bar:<10}]")

    # --- 5. GÉNÉRATION INTELLIGENTE (TRI PAR SCORE) ---
    # On crée une liste de paires (Lettre, Score)
    candidates = []
    for idx, score in enumerate(results):
        if score >= 90.0:
            candidates.append((CHAR_LABELS[idx], score))
    
    # ON TRIE pour avoir les meilleurs en premier (Score décroissant)
    # C'est la ligne qui change tout !
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # On prend les 4 premiers du classement
    if len(candidates) >= 4:
        top_4 = candidates[:4]
        suggested_code = [c[0] for c in top_4] # On garde juste la lettre
        hybrid_code_str = " - ".join(suggested_code)
    elif len(candidates) > 0:
        suggested_code = [c[0] for c in candidates]
        hybrid_code_str = " - ".join(suggested_code)
    else:
        hybrid_code_str = "AUCUN (Scores trop faibles)"

    print("\n" + "="*40)
    print(f" PROPOSITION DE CODE HYBRIDE OPTIMISÉ ")
    print("="*40)
    print(f"Meilleurs candidats : {candidates}") # Affiche le classement
    print(f"CODE SUGGÉRÉ POUR {TARGET_ID} :  >> {hybrid_code_str} <<")
    print("="*40 + "\n")

    # --- 6. GRAPHIQUE FINAL ---
    plt.figure(figsize=(15, 7))
    
    # Couleurs : Vert (>90), Rouge (<70), Bleu (Entre deux)
    colors = []
    for x in results:
        if x >= 90: colors.append('#2ecc71') 
        elif x < 70: colors.append('#e74c3c') 
        else: colors.append('#3498db') 
        
    bars = plt.bar(range(62), results, color=colors, edgecolor='black', alpha=0.8)
    
    plt.axhline(y=90, color='#27ae60', linestyle='--', linewidth=2, label='Seuil Excellence (Vert)')
    plt.axhline(y=70, color='#c0392b', linestyle=':', linewidth=2, label='Seuil Critique (Rouge)')
    
    # Annotation du Code Hybride
    plt.text(31, 105, f"CODE OPTIMAL : {hybrid_code_str}", 
             ha='center', va='bottom', fontsize=14, fontweight='bold', 
             color='white', bbox=dict(facecolor='#2c3e50', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.xticks(range(62), CHAR_LABELS, rotation=90, fontsize=8)
    plt.title(f'Analyse de Discriminabilité - Sujet {TARGET_ID}\n(Sélection des 4 Meilleurs Scores)', fontsize=14)
    plt.ylabel('Précision Test (%)')
    plt.ylim(0, 115) 
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    output_name = f"Analyse_Discrimination_{TARGET_ID}.png"
    plt.savefig(output_name)
    print(f"[SUCCES] Graphique sauvegardé : {output_name}")
    plt.show()