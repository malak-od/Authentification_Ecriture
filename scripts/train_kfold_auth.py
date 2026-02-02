# -*- coding: utf-8 -*-
import os
import sys
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

# --- 1. CONFIGURATION ---
DATA_DIR = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete"
TARGET_ID = '002'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import des modules locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from snn_engine import AuthSNN, rate_coding
from utils import compute_dynamics

# --- 2. DATASET INTELLIGENT (Sépare les Séries) ---
class RigorousDataset(Dataset):
    def __init__(self, root_dir, target_id, mode='train', length=20):
        """
        mode: 'train' (Séries 1, 2, 3, 4) ou 'test' (Série 5)
        """
        self.mode = mode
        self.samples = []
        target_samples = []
        imposter_samples = []
        
        if not os.path.exists(root_dir):
            print(f"[ERREUR] Dossier introuvable : {root_dir}")
            return

        files = glob.glob(os.path.join(root_dir, '*_preprocessed'))
        files = [f for f in files if not f.endswith('_info')]
        
        print(f"Chargement des données ({mode.upper()})...")

        for filepath in files:
            filename = os.path.basename(filepath)
            is_target = filename.startswith(target_id)
            label = 1.0 if is_target else 0.0
            
            try:
                with open(filepath, 'r') as f: lines = f.readlines()
            except: continue

            # On parcourt les 62 caractères
            for char_idx in range(62):
                start = char_idx * 10
                end = start + 10
                if len(lines) < end: continue
                
                # Bloc de 10 lignes = 5 essais (Donnée + Label)
                # Les données sont aux lignes 0, 2, 4, 6, 8 dans ce bloc
                block = lines[start:end]
                data_lines = block[0::2] # indices 0, 1, 2, 3, 4
                
                # SÉLECTION DES SÉRIES (C'est ici que ça se joue)
                if mode == 'train':
                    selected_lines = data_lines[0:4] # On garde les 4 premiers essais
                else: # mode == 'test'
                    selected_lines = [data_lines[4]] # On garde UNIQUEMENT le 5ème essai

                for line in selected_lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    try:
                        vals = np.array([float(x) for x in parts])
                        num_points = len(vals) // 5
                        traj = vals[:num_points*5].reshape(-1, 5)
                        feat = compute_dynamics(traj)
                        if len(feat) == 0: continue
                        
                        # Padding / Cutting
                        final_seq = None
                        if len(feat) < length:
                            if len(feat) > 5:
                                pad = length - len(feat)
                                final_seq = np.pad(feat, ((0, pad), (0,0)), mode='edge')
                        else:
                            final_seq = feat[:length]
                        
                        if final_seq is not None:
                            item = (torch.FloatTensor(final_seq), torch.tensor(label).float())
                            if is_target: target_samples.append(item)
                            else: imposter_samples.append(item)
                    except: continue

        # --- SUR-ÉCHANTILLONNAGE (Uniquement pour le Train) ---
        # Pour le Test, on laisse tel quel (pour voir la vraie performance en conditions réelles)
        if mode == 'train':
            if len(target_samples) > 0 and len(imposter_samples) > 0:
                factor = len(imposter_samples) // len(target_samples)
                balanced_targets = target_samples * factor
                remainder = len(imposter_samples) % len(target_samples)
                balanced_targets += target_samples[:remainder]
                self.samples = balanced_targets + imposter_samples
                random.shuffle(self.samples)
            else:
                self.samples = []
        else:
            # Pour le test, on garde tout le monde (déséquilibré, c'est la vie réelle)
            self.samples = target_samples + imposter_samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        
        # --- SOLUTION ANTI-PAR-CŒUR : DATA AUGMENTATION ---
        # Si on est en train d'apprendre (pas en test), on ajoute un mini bruit
        # Cela empêche le réseau de mémoriser les valeurs exactes.
        if self.mode == 'train':
            noise = torch.randn_like(seq) * 0.05  # 5% de bruit gaussien
            seq = seq + noise
            
        return seq, label

# --- 3. FONCTIONS D'ENTRAINEMENT ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    from spikingjelly.activation_based import functional
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(rate_coding(x)).mean(0).squeeze(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    from spikingjelly.activation_based import functional
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(rate_coding(x)).mean(0).squeeze(-1)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
            functional.reset_net(model)
    return 100 * correct / total

# --- 4. MAIN ---
if __name__ == "__main__":
    print(f"--- PROTOCOLE RIGOUREUX (DEMANDE PATRICK) ---")
    
    # A. Préparation des données
    # Train = Séries 1, 2, 3, 4 (Avec Oversampling)
    train_ds = RigorousDataset(DATA_DIR, TARGET_ID, mode='train')
    # Test = Série 5 (Sans triche)
    test_ds = RigorousDataset(DATA_DIR, TARGET_ID, mode='test')
    
    print(f"Taille Train (Séries 1-4 équilibrées) : {len(train_ds)}")
    print(f"Taille Test  (Série 5 brute)        : {len(test_ds)}")

    # B. Phase 1 : Cross-Validation (4-Fold) sur le Train
    print("\n--- PHASE 1 : CROSS-VALIDATION (4-FOLD) ---")
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_ds)):
        print(f"Fold {fold+1}/4...", end="")
        
        train_sub = Subset(train_ds, train_idx)
        val_sub = Subset(train_ds, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
        
        model = AuthSNN(num_inputs=9).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # On entraine 15 epochs
        for _ in range(100):
            train_one_epoch(model, train_loader, optimizer, criterion)
            
        acc = evaluate(model, val_loader)
        fold_results.append(acc)
        print(f" Val Acc: {acc:.2f}%")

    avg_val = sum(fold_results) / 4
    print(f"MOYENNE VALIDATION (4-Fold) : {avg_val:.2f}%")

    # C. Phase 2 : Test Final (Hold-Out)
    print("\n--- PHASE 2 : TEST FINAL (SÉRIE 5) ---")
    # On ré-entraîne sur TOUT le dataset train
    full_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    final_model = AuthSNN(num_inputs=9).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("Ré-entraînement complet...", end="")
    for epoch in range(100): # Un peu plus long pour le final
        train_one_epoch(final_model, full_train_loader, optimizer, criterion)
    print(" Terminé.")
    
    final_acc = evaluate(final_model, test_loader)
    print(f"\nRESULTAT FINAL (Série 5 jamais vue) : {final_acc:.2f}%")
    
    if final_acc > 90:
        print("CONCLUSION : Le système est TRÈS ROBUSTE et généralise bien.")
    elif final_acc > 80:
        print("CONCLUSION : Le système est BON, mais perfectible.")
    else:
        print("CONCLUSION : Le système apprend par coeur (Overfitting).")