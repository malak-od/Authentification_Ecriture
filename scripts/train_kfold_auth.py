import os
import sys
import glob
import torch
from torch.utils.data import Dataset

# Import du moteur commun et des utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_digilets_file, compute_dynamics
from snn_engine import run_kfold_training # <--- L'IMPORT MAGIQUE

# Configuration
DATA_DIR = r"C:\Users\User\OneDrive\Desktop\PFE\Authentification_Ecriture\data\preprocessed\complete"
TARGET_ID = '002'

class DigiLeTsAuthDataset(Dataset):
    def __init__(self, root_dir, target_id, length=20):
        self.samples = []
        files = glob.glob(os.path.join(root_dir, '*_preprocessed'))
        files = [f for f in files if not f.endswith('_info')]
        
        print(f" Chargement Dataset COMPLET (Lettres + Chiffres)...")
        
        for filepath in files:
            filename = os.path.basename(filepath)
            label = 1.0 if filename.startswith(target_id) else 0.0
            
            raw_matrix = read_digilets_file(filepath)
            if raw_matrix is not None:
                features = compute_dynamics(raw_matrix)
                
                # Équilibrage : on limite un peu les imposteurs
                num_seq = len(features) // length
                if label == 0.0 and num_seq > 5: num_seq = 5
                
                for i in range(num_seq):
                    seq = features[i*length : (i+1)*length]
                    self.samples.append((torch.FloatTensor(seq), torch.tensor(label).float()))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

if __name__ == "__main__":
    # 1. On crée le dataset
    ds = DigiLeTsAuthDataset(DATA_DIR, TARGET_ID)
    
    # 2. On lance le moteur K-Fold
    run_kfold_training(ds, k=5, epochs=30, lr=0.005)