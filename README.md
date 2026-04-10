# 🫀 Cardiac MRI Segmentation — U-Net Deep Learning

> Segmentation automatique de l'endocarde sur images IRM cardiaques par Deep Learning  
> **Université Clermont Auvergne** · Master 2 TSI-ITM · 2024–2025  
> Encadrant : Dr. Omar Aït Aider

---

## 📌 Description

Ce projet implémente un modèle de **segmentation sémantique** de l'endocarde sur des images IRM cardiaques en coupe petit axe, en utilisant une architecture **U-Net** entraînée avec TensorFlow/Keras.

L'objectif est de produire automatiquement des masques de segmentation précis, comparables aux annotations manuelles réalisées par des experts cliniques, afin d'assister le diagnostic cardiologique.

---

## 🧠 Architecture

```
Input IRM (2D)
     │
  Encoder (Contracting Path)
  ├── Conv2D + BN + ReLU  (×2)
  ├── MaxPooling2D
  └── ... (4 niveaux)
     │
  Bottleneck
     │
  Decoder (Expansive Path)
  ├── UpSampling2D
  ├── Skip Connections
  └── Conv2D + BN + ReLU  (×2)
     │
Output Mask (sigmoid)
```

**Architecture** : U-Net 2D  
**Framework** : TensorFlow 2.x / Keras  
**Fonction de perte** : Hybride — Dice Loss + Binary Crossentropy  
**Optimiseur** : Adam  
**Accélération** : GPU (Google Colab)

---

## ⚙️ Méthodologie

### Prétraitement
- Normalisation des intensités (min-max)
- Redimensionnement des images en 256×256
- Conversion des masques en binaire (endocarde / fond)

### Entraînement
- **Data Augmentation** : rotations, flips horizontaux/verticaux, zoom aléatoire
- **EarlyStopping** : patience = 15 époques sur la val_loss
- **ReduceLROnPlateau** : réduction du learning rate si stagnation

### Évaluation
- **Coefficient de Dice** (métrique principale)
- Comparaison avec les annotations manuelles d'experts

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Dice Score (validation) | **~0.87** |
| Loss finale | converge |
| Robustesse | Data Augmentation + EarlyStopping |

---

## 🛠️ Stack Technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## 📁 Structure du projet

```
cardiac-mri-segmentation/
│
├── model/
│   └── unet_model.py          # Architecture U-Net
│
├── training/
│   ├── train.py               # Script d'entraînement
│   └── losses.py              # Dice Loss + BCE hybride
│
├── evaluation/
│   └── metrics.py             # Calcul du coefficient de Dice
│
├── notebooks/
│   └── cardiac_segmentation.ipynb   # Pipeline complet
│
├── results/
│   └── figures/               # Courbes de loss, exemples de segmentation
│
└── README.md
```

---

## 🚀 Utilisation

```python
# Cloner le dépôt
git clone https://github.com/ghizlane-echchiguer/cardiac-mri-segmentation.git

# Installer les dépendances
pip install tensorflow keras numpy opencv-python matplotlib

# Lancer le notebook
jupyter notebook notebooks/cardiac_segmentation.ipynb
```

---

## 🔗 Contexte académique

| | |
|---|---|
| **Formation** | Master 2 Traitement du Signal et des Images – Imagerie et Technologie pour la Médecine |
| **Établissement** | École Universitaire de Physique et d'Ingénierie (EUPI), Clermont-Ferrand |
| **Année** | 2024 – 2025 |
| **Encadrant** | Dr. Omar Aït Aider, Université Clermont Auvergne |

---

## 👩‍💻 Auteure

**Ghizlane Ech-chiguer**  
Ingénieure Biomédicale · M2 TSI-ITM  
[LinkedIn](https://www.linkedin.com/in/ghizlaneechchiguer/) · [GitHub](https://github.com/ghizlane-echchiguer)

---

*Projet académique — Master 2 · 2024–2025*
