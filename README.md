# Normalizing Flows Implementation

Ce projet est une implémentation interactive de différents modèles de Normalizing Flows (Flux Normalisants) avec une interface utilisateur Streamlit. Il permet de visualiser et d'expérimenter avec les transformations probabilistes NICE, RealNVP et Glow sur différentes distributions de données.

## 📋 Table des Matières

- [Introduction](#introduction)
- [Modèles Implémentés](#modèles-implémentés)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Aspects Techniques](#aspects-techniques)
- [Contributions](#contributions)
- [Références](#références)

## 🔍 Introduction

Les Normalizing Flows sont des modèles génératifs qui apprennent à transformer une distribution simple (comme une gaussienne) en une distribution complexe à travers une séquence de transformations inversibles. Ce projet offre une interface interactive pour explorer ces modèles, comprendre leur fonctionnement et visualiser leurs transformations.

## 🧠 Modèles Implémentés

### NICE (Non-linear Independent Components Estimation)
- Utilise des couplages additifs pour transformer les distributions
- Architecture simplifiée avec des transformations inversibles explicites
- Implémentation basée sur l'article ["NICE: Non-linear Independent Components Estimation"](https://arxiv.org/abs/1410.8516)

### RealNVP (Real-valued Non-Volume Preserving)
- Extension de NICE avec des transformations affines (multiplication et addition)
- Permet des mappings plus expressifs grâce aux changements d'échelle
- Basé sur l'article ["Density Estimation using Real NVP"](https://arxiv.org/abs/1605.08803)

### Glow
- Architecture avancée combinant convolutions 1x1 inversibles et couplages affines
- Inclut une normalisation par lots et des réseaux plus profonds
- Implémentation selon l'article ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)

## 💻 Installation

```bash
# Cloner le repository
git clone https://github.com/EpsilonOF/ProjetDL.git
cd ProjetDL

# Créer et activer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

Pour lancer l'application Streamlit:

```bash
streamlit run app.py
```

L'interface web devrait s'ouvrir automatiquement dans votre navigateur, vous permettant de:
- Sélectionner un modèle (NICE, RealNVP, Glow)
- Choisir une distribution de données (Cercles, Spirale, Gaussiennes multiples, etc.)
- Ajuster les hyperparamètres d'entraînement
- Visualiser les transformations en temps réel

## 📁 Structure du Projet

```
ProjetDL/
├── app.py                  # Point d'entrée de l'application Streamlit
├── requirements.txt        # Dépendances du projet
├── presentation.py         # Présentation du sujet
├── nice/                   # Implémentation du modèle NICE
│   ├── nice.py             # Mise en page du modèle
│   ├── train_nice.py       # Entraînement du modèle
│   └── images/             # Si besoin d'enregistrer des images
├── realnvp/                # Implémentation du modèle RealNVP
│   ├── real_nvp.py         # Mise en page du modèle
│   ├── train_realnvp.py    # Entraînement du modèle
│   └── images/             # Si besoin d'enregistrer des images
├── glow/                   # Implémentation du modèle Glow
│   ├── glow.py             # Mise en page du modèle
│   ├── train_glow.py       # Entraînement du modèle
│   └── images/             # Si besoin d'enregistrer des images
```

## ✨ Fonctionnalités

- **Visualisation en temps réel**: Observez comment la distribution se transforme pendant l'entraînement
- **Personnalisation des hyperparamètres**: Ajustez le taux d'apprentissage, la taille des lots, le nombre d'epochs, etc.
- **Analyse comparative**: Comparez les performances des différents modèles sur les mêmes données

## 🔧 Aspects Techniques

### Architecture des modèles

Les modèles sont implémentés comme des modules PyTorch héritant d'une classe de base commune, ce qui permet une interface cohérente pour:
- La transformation directe (forward) d'une distribution simple en distribution complexe
- La transformation inverse (backward) pour générer de nouveaux échantillons
- Le calcul du log-déterminant jacobien pour l'estimation de densité

### Entraînement

L'entraînement utilise la maximisation de la vraisemblance (maximum likelihood estimation) comme objectif:
- Minimiser la divergence KL entre la distribution cible et la distribution transformée
- Optimisation par descente de gradient stochastique avec Adam
- Suivi des métriques d'entraînement comme la log-vraisemblance négative

## 🤝 Contributions

Les contributions sont les bienvenues! Pour contribuer:
1. Forkez le repository
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📚 Références

- [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803)
- [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)
