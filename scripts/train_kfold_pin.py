import os
import sys
import glob
import torch
from torch.utils.data import Dataset

# Import du moteur commun
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_digilets_file, compute_dynamics
from snn_engine import run_kfold_training # <--- L'IMPORT MAGIQUE

# Configuration
DATA_DIR = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete"
TARGET_ID = '002'

class DigiLeTsPinDataset(Dataset):
    def __init__(self, root_dir, target_id, length=20):
        self.samples = []
        count_target = 0
        
        # Configuration Filtre PIN
        TOTAL_TRAJS = 310
        MAX_TRAJECTORIES = 50 
        
        files = glob.glob(os.path.join(root_dir, '*_preprocessed'))
        files = [f for f in files if not f.endswith('_info')]
        
        print(f" Chargement Dataset CODE PIN (Chiffres uniquement)...")
        
        for filepath in files:
            filename = os.path.basename(filepath)
            is_target = filename.startswith(target_id)
            label = 1.0 if is_target else 0.0
            
            raw_data = read_digilets_file(filepath)
            if raw_data is not None:
                # --- FILTRE SPECIAL PIN ---
                ratio = MAX_TRAJECTORIES / TOTAL_TRAJS
                limit_index = int(len(raw_data) * ratio)
                digits_data = raw_data[:limit_index]
                
                features = compute_dynamics(digits_data)
                
                num_seq = len(features) // length
                if not is_target and num_seq > 5: num_seq = 5
                
                for i in range(num_seq):
                    seq = features[i*length : (i+1)*length]
                    self.samples.append((torch.FloatTensor(seq), torch.tensor(label).float()))
                    if is_target: count_target += 1

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

if __name__ == "__main__":
    # 1. On crée le dataset PIN
    ds = DigiLeTsPinDataset(DATA_DIR, TARGET_ID)
    
    # 2. On lance le moteur K-Fold
    run_kfold_training(ds, k=5, epochs=30, lr=0.005)