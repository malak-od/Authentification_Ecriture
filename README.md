# Authentification_Ecriture
Authentification Biométrique de l'Écriture Manuscrite par SNN (Spiking Neural Networks)

Ce dépôt contient le code source et les outils de traitement de données pour mon Projet de Fin d'Études (PFE). 
L'objectif est de développer un système d'authentification biométrique basé sur la dynamique de l'écriture manuscrite, en utilisant des architectures éco-efficientes de type **Réseaux de Neurones Impulsionnels (SNN)**.

##  Contexte du Projet

L'authentification par signature ou écriture manuscrite se divise en deux catégories : statique (image) et dynamique (trajectoire). Ce projet se concentre sur l'approche **dynamique**, qui capture l'évolution temporelle du mouvement (position, pression, vitesse, accélération), rendant la falsification extrêmement difficile.

Nous utilisons le framework **SpikingJelly** (basé sur PyTorch) pour exploiter la nature temporelle et événementielle des données via des SNNs.

##  Description de la Base de Données (DigiLeTs)

Les données utilisées proviennent du projet de recherche **DigiLeTs** (Naturally Handwritten Character Trajectories).

* **Source Originale :** [GitHub DigiLeTs](https://github.com/CognitiveModeling/DigiLeTs)
* **Nature des Données :** Séries temporelles capturées via une tablette Wacom Intuos.
* **Participants :** Données collectées auprès de multiples participants (identifiés par ID, âge, genre, latéralité).
* **Volume :** Chaque participant a rédigé **5 instances** de **62 caractères** (0-9, a-z, A-Z), soit 310 trajectoires par personne.

### Format des Données
Chaque fichier de données représente un participant. Les trajectoires sont stockées sous forme de séquences de points. Pour ce projet, nous exploitons les caractéristiques brutes suivantes :

1.  **Coordonnée X** : Position horizontale normalisée.
2.  **Coordonnée Y** : Position verticale normalisée.
3.  **Pression** : Force appliquée par le stylet (caractéristique biométrique clé).
4.  **État du Stylet (Pen Down)** : Indicateur binaire (1 = posé, 0 = levé).
5.  **Timestamp** : Information temporelle relative.

> **Note Technique :** En plus de ces données brutes, le scripts (`digilets_dataset.py`) calculent automatiquement les dérivées dynamiques (Vitesse $V_x, V_y$, Magnitude de Vitesse, et Accélération) pour enrichir l'entrée du réseau neuronal.

##  Scripts de Visualisation et Analyse

Ce dépôt inclut des outils Python développés pour explorer et valider la qualité des données biométriques.

### 1. Visualisation Globale des Caractères (`scripts/generate_visualizations.py`)
Ce script génère une planche contact (grille) visualisant l'intégralité des 310 caractères écrits par un participant.
* **Entrée :** Fichiers de données brutes du dossier `data/`.
* **Sortie :** Image `.png` montrant la forme et la pression (épaisseur du trait) de chaque essai.
* **Usage :**
    ```bash
    python scripts/generate_visualizations.py
    ```

### 2. Analyse de la Dynamique (`scripts/visualize_dynamics.py`)
Ce script permet d'analyser la "signature dynamique" d'un individu pour un caractère donné. Il superpose les 5 essais d'un même caractère pour visualiser la régularité du scripteur.
* **Fonctionnalité :** Trace les courbes de **Magnitude de Vitesse** et de **Pression** en fonction du temps normalisé.
* **Indicateurs :** Affiche la courbe moyenne (signature) et l'écart-type (variabilité intra-classe).
* **Usage :**
    ```bash
    python scripts/visualize_dynamics.py
    ```

### 3. Loader PyTorch (`scripts/digilets_dataset.py`)
Classe `Dataset` compatible avec PyTorch et SpikingJelly. Elle gère :
* Le chargement robuste des fichiers textes.
* Le nettoyage des données (suppression des métadonnées).
* Le calcul des dérivées (Vitesse/Accélération).
* La normalisation temporelle pour l'entraînement des SNN.

##  Installation et Configuration de l'Environnement
Ce projet a été développé dans un environnement virtuel **Anaconda** pour garantir la gestion des dépendances (notamment pour l'accélération GPU avec CUDA).
### Pré-requis
* [Anaconda](https://www.anaconda.com/) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installé sur votre machine.
* Visual Studio Code (recommandé).

### 1. Création de l'environnement virtuel
Nous utilisons Python 3.10 pour une compatibilité optimale avec SpikingJelly. Ouvrez votre terminal (Anaconda Prompt ou VS Code) et exécutez :

```bash
# Création de l'environnement nommé 'pfe_auth'
conda create -n pfe_auth python=3.10

# Activation de l'environnement
conda activate pfe_auth
````

##  Références

* **DigiLeTs Project**: *Naturally handwritten character trajectories*. [Lien GitHub](https://github.com/CognitiveModeling/DigiLeTs).
* **SpikingJelly**: *A deep learning framework for Spiking Neural Networks*. [Lien GitHub](https://github.com/fangwei123456/spikingjelly?tab=readme-ov-file).
