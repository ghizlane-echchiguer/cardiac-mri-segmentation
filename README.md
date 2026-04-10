# 🫀 Cardiac MRI Segmentation — U-Net Deep Learning

> Segmentation automatique de l'endocarde sur images IRM cardiaques par Deep Learning  
> **Université Clermont Auvergne** · Master 2 TSI-ITM · 2024–2025  
> Encadrant : Dr. Omar Aït Aider

---

## 📌 Description

Ce projet implémente un modèle de **segmentation sémantique de l'endocarde** sur des images IRM cardiaques 2D en utilisant une architecture **U-Net** entraînée avec TensorFlow/Keras.

L'objectif est de remplacer les méthodes manuelles de segmentation — coûteuses en temps — par une approche automatisée capable de produire des masques précis, comparables aux annotations d'experts cliniques.

---

## 🗂️ Structure du projet

```
cardiac-mri-segmentation/
│
├── model.py                    # Architecture U-Net complète
├── metrics_and_losses.py       # Dice Loss + Binary Crossentropy hybride
├── TrainUNet.ipynb              # Pipeline d'entraînement complet
├── PredictWithUnet.ipynb        # Inférence et évaluation des résultats
├── report/
│   └── rapport.pdf              # Rapport complet du projet
└── README.md
```

---

## 🧠 Architecture U-Net

```
Input IRM 2D (256×256, grayscale)
        │
   ┌────▼────────────────────────────────┐
   │  ENCODEUR (Downsampling)            │
   │  Conv2D(32) → Pool                  │
   │  Conv2D(64) → Pool                  │
   │  Conv2D(128) → Pool                 │
   │  Conv2D(256) → Pool                 │
   └────────────────┬────────────────────┘
                    │
   ┌────────────────▼────────────────────┐
   │  BOTTLENECK — Conv2D(512)           │
   └────────────────┬────────────────────┘
                    │
   ┌────────────────▼────────────────────┐
   │  DÉCODEUR (Upsampling)              │
   │  ConvTranspose + Skip Connections   │
   │  Conv2D(256) → Conv2D(128)          │
   │  Conv2D(64)  → Conv2D(32)           │
   └────────────────┬────────────────────┘
                    │
         Output Mask — Conv2D(1, sigmoid)
```

| Paramètre | Valeur |
|---|---|
| Framework | TensorFlow 2.x / Keras |
| Taille d'entrée | 256 × 256 px (niveaux de gris) |
| Activation sortie | Sigmoid |
| Optimiseur | Adam |
| Fonction de perte | Dice Loss + Binary Crossentropy (pondération 50/50) |
| Epochs | 50 |
| Accélération | GPU (Google Colab) |

---

## ⚙️ Méthodologie

### 1. Données
- **330 images IRM 2D** annotées manuellement (endocarde)
- Format `.png` — Images : `/frames` · Masques : `/masks_endo`
- Split : **300 train / 30 validation**

### 2. Prétraitement
- Normalisation des pixels entre 0 et 1
- Redimensionnement à 256 × 256 px

### 3. Data Augmentation (entraînement uniquement)
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

### 4. Callbacks
- **ModelCheckpoint** — sauvegarde des meilleurs poids
- **EarlyStopping** — arrêt si la val_loss ne s'améliore plus

---

## 📊 Résultats

| Métrique | Entraînement | Validation |
|---|---|---|
| **Dice Coefficient** | **0.507** | **0.549** |
| Distance Euclidienne | — | 6.84 |
| Distance Hausdorff | — | 47.86 |

> Les masques prédits montrent une bonne correspondance visuelle avec les annotations manuelles des experts cliniques (voir rapport pour figures).

---

## 🛠️ Stack Technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

## 🚀 Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/ghizlane-echchiguer/cardiac-mri-segmentation.git
cd cardiac-mri-segmentation

# Installer les dépendances
pip install tensorflow keras numpy opencv-python matplotlib scikit-image scipy
```

```python
# Entraînement — ouvrir le notebook
jupyter notebook TrainUNet.ipynb

# Prédiction et évaluation
jupyter notebook PredictWithUnet.ipynb
```

> ⚠️ **Note** : Le dataset (images IRM) n'est pas inclus dans ce dépôt pour des raisons de confidentialité médicale. Les fichiers de poids `.h5` ne sont pas inclus en raison de leur taille.

---

## 📄 Rapport

Le rapport complet du projet est disponible dans [`report/rapport.pdf`](report/rapport.pdf).  
Il détaille la méthodologie, l'architecture, les résultats chiffrés et les visualisations des masques prédits.

---

## 🔗 Contexte académique

| | |
|---|---|
| **Formation** | Master 2 Traitement du Signal et des Images – Imagerie et Technologie pour la Médecine |
| **Établissement** | École Universitaire de Physique et d'Ingénierie (EUPI) — Université Clermont Auvergne |
| **Année** | 2024 – 2025 |
| **Encadrant** | Dr. Omar Aït Aider |

---

## 👩‍💻 Auteure

**Ghizlane Ech-chiguer**  
Ingénieure Biomédicale · M2 TSI-ITM  
[LinkedIn](https://www.linkedin.com/in/ghizlaneechchiguer/) · [GitHub](https://github.com/ghizlane-echchiguer)

---

*Projet académique — Master 2 · Université Clermont Auvergne · 2024–2025*
