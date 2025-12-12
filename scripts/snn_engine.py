import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# === 1. LE CERVEAU (Modèle SNN) ===
class AuthSNN(nn.Module):
    def __init__(self, num_inputs=9):
        super().__init__()
        self.net = nn.Sequential(
            layer.Linear(num_inputs, 64),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            
            layer.Linear(64, 32),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
            
            layer.Linear(32, 1) 
        )

    def forward(self, x):
        x = x.permute(1, 0, 2) # [Batch, T, F] -> [T, Batch, F]
        return functional.multi_step_forward(x, self.net)

def rate_coding(data):
    return (data > torch.rand_like(data)).float()

# === 2. LA MÉTHODE D'EXAMEN (K-Fold Générique) ===
def run_kfold_training(dataset, k=5, epochs=30, batch_size=16, lr=0.005, device=None):
    """
    Fonction universelle qui prend N'IMPORTE QUEL dataset et fait une validation croisée.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Préparation du découpage
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []
    
    print(f"\n Démarrage de la Validation Croisée ({k}-Fold)")
    print(f"   - Device: {device}")
    print(f"   - Échantillons totaux: {len(dataset)}")
    print(f"   - Époques par Fold: {epochs}")
    print("-" * 50)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f" FOLD {fold+1}/{k}")
        
        # 1. On coupe les données
        train_sub = SubsetRandomSampler(train_ids)
        test_sub = SubsetRandomSampler(test_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sub)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sub)
        
        # 2. On réinitialise un NOUVEAU cerveau vierge (Crucial !)
        model = AuthSNN(num_inputs=9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # 3. Boucle d'entraînement
        for epoch in range(epochs):
            model.train()
            for features, label in train_loader:
                features, label = features.to(device), label.to(device)
                optimizer.zero_grad()
                
                spikes = rate_coding(features)
                output = model(spikes)
                score = output.mean(0).squeeze(-1)
                
                loss = criterion(score, label)
                loss.backward()
                optimizer.step()
                functional.reset_net(model)

        # 4. Test Final du Fold
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, label in test_loader:
                features, label = features.to(device), label.to(device)
                spikes = rate_coding(features)
                output = model(spikes)
                score = output.mean(0).squeeze(-1)
                
                preds = (torch.sigmoid(score) > 0.5).float()
                correct += (preds == label).sum().item()
                total += label.size(0)
                functional.reset_net(model)
        
        acc = 100 * correct / total
        print(f"    Résultat : {acc:.2f}%")
        results.append(acc)

    # Résultat Final
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    print("=" * 50)
    print(f" PERFORMANCE FINALE (Moyenne) : {mean_acc:.2f}% (+/- {std_acc:.2f})")
    print("=" * 50)