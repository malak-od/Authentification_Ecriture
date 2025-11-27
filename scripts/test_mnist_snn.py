import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# --- CONFIGURATION ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 #0.001 Taux d'apprentissage
T = 10  # Nombre de pas de temps (Time Steps)
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Utilisation du périphérique : {DEVICE}")

# --- 1. PRÉPARATION DES DONNÉES (MNIST) ---
# On transforme les images en Tenseurs
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  #Découpe les données en "lots" (batches) de 64 images.
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --- 2. MODÈLE SNN (Convolutionnel Simple) ---
class CSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Le réseau : Conv -> LIF -> MaxPool -> Conv -> LIF -> MaxPool -> Flatten -> Linear -> LIF
        self.net = nn.Sequential(     #Empile les couches les unes après les autres
            layer.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            # Neurone LIF avec gradient "ATan" (très stable)
            neuron.LIFNode(surrogate_function=surrogate.ATan()),   #remplace la fonction d'activation classique par une fonction de type LIF (Leaky Integrate-and-Fire) avec une fonction de substitution ATan pour le gradient.
            layer.MaxPool2d(2, 2),  # 14x14

            layer.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7x7

            layer.Flatten(),
            layer.Linear(128 * 7 * 7, 128, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            
            layer.Linear(128, 10, bias=False), # Sortie : 10 classes
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        # x est une image statique [Batch, 1, 28, 28]
        # On la répète T fois pour simuler une entrée temporelle (Direct Coding)
        # Entrée SNN : [T, Batch, 1, 28, 28]
        x_seq = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # x_seq devient [T, Batch, 1, 28, 28] (Séquence vidéo statique)
        
        # functional.multi_step_forward gère la dimension T automatiquement
        return functional.multi_step_forward(x_seq, self.net)

# --- 3. BOUCLE D'ENTRAÎNEMENT ---
model = CSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Démarrage de l'entraînement pour {EPOCHS} époques...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_acc = 0
    total_samples = 0

    for img, label in train_loader:
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward : Le modèle tourne sur T pas de temps
        output = model(img) # Shape: [T, Batch, 10]
        
        # Décodage : On fait la moyenne des spikes de sortie sur le temps (Firing Rate)
        output_mean = output.mean(0) 
        
        loss = F.cross_entropy(output_mean, label)
        loss.backward()
        optimizer.step()
        
        # CRUCIAL : Reset des neurones (vider le potentiel de membrane)
        functional.reset_net(model)

        # Calcul stats
        train_loss += loss.item() * label.numel()
        train_acc += (output_mean.argmax(1) == label).float().sum().item()
        total_samples += label.numel()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/total_samples:.4f} | Acc: {train_acc/total_samples*100:.2f}%")

print("Terminé ! ")