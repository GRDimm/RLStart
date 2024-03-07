# RLStart

## Repository de Projets de Reinforcement Learning

Ce repository contient des petits projets de reinforcement learning que je développe seul pour apprendre et explorer différentes techniques et algorithmes dans le domaine.

## Bibliothèques Utilisées
- `gymnasium`: Pour la création et la manipulation d'environnements de reinforcement learning.
- `stable_baselines3`: Une bibliothèque offrant des implémentations de qualité de divers algorithmes de reinforcement learning.
- `box2d`: Utilisé pour certains environnements nécessitant la physique, comme le Lunar Lander.

## Notions

### Utilisation de PPO
Implémentation de l'algorithme Proximal Policy Optimization (PPO) sur différents environnements Gymnasium.

### Utilisation de DQN
Mise en œuvre de Deep Q-Network (DQN) pour apprendre des stratégies optimales dans des environnements classiques.

### Création d'un environnement custom de Lunar Lander
Développement et entraînement d'un modèle dans un environnement personnalisé de Lunar Lander, visant à tester et améliorer les compétences de conception d'environnements.

## Projets/Fichiers

### 0 - Lunar Lander V2
- Environement LunarLander-v2 de gym.
- Objectif : comprendre comment entrainer un modèle avant de coder l'environement soi-même.

### 1 - CartPole 
- Environement custom reproduisant l'environement CartPole-v1 de gym.
- Objectif : bouger de droite à gauche pour ne par faire tomber la barre

### 2 - Custom Lander 
- Environement custom inspiré de l'environement LunarLander-v2 de gym.
- Objectif : Aterrir proprement sur une plateforme positionnée aléatoirement avec un moteur central et deux moteurs latéraux.

<p align="center">
  <img src="https://raw.githubusercontent.com/GRDimm/RLStart/main/images/CustomLander.gif" width="80%" height="80%" />
</p>

### 3 - Custom Explorer
- Environement custom tiré de Custom Lander.
- Objectif : Explorer la map, rewards d'exploration et de découverte de l'objectif. 

## Installation
- Environement Anaconda, Python >= 3.8
- Installer les modules de `requirements.txt`