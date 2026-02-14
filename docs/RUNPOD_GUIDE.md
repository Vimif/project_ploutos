# 🌩️ RunPod / Cloud Training Guide

Ce guide explique comment déployer Ploutos sur une instance GPU (RunPod, Lambda Labs, Vast.ai) pour diviser le temps d'entraînement par 10.

## 1. Pré-requis

### A. Clé SSH (sur votre PC local)
Si vous n'en avez pas :
```powershell
ssh-keygen -t ed25519 -C "runpod-key"
# Copier la clé publique affichée ou dans ~/.ssh/id_ed25519.pub
```

### B. Compte RunPod
*   Créer un compte sur [RunPod.io](https://www.runpod.io).
*   Ajouter ~$10-20 de crédits.
*   Ajouter votre clé SSH dans **Settings > SSH Keys**.

---

## 2. Déploiement de l'Instance

1.  Aller sur **Secure Cloud**.
2.  Choisir un GPU : **RTX 3090** (24GB VRAM) ou **A5000** (~$0.40/h).
3.  Choisir le Template : **RunPod Pytorch 2.1** (ou 2.0+).
4.  Cocher "Start Jupyter Lab".
5.  Cliquer sur **Deploy**.

---

## 3. Connexion & Installation

1.  Connectez-vous via SSH (commande donnée par RunPod, ex: `ssh root@x.x.x.x -p 12345`).
2.  Dans le terminal du Pod, lancez :

```bash
# 1. Cloner le repo
git clone https://github.com/Vimif/project_ploutos.git
cd project_ploutos

# 2. Lancer le setup automatique (2 min)
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

> **Note :** Le script installe tout, crée les dossiers, et génère des alias pour lancer les entraînements.

---

## 4. Lancer les Entraînements

> **Important :** Utiliser la config cloud (`training_config_v8_cloud.yaml`) qui exploite le GPU avec 32 envs parallèles au lieu de 8.

### A. Baseline (PPO Standard)
```bash
python training/train_walk_forward.py --config config/training_config_v8_cloud.yaml
```

### B. Recurrent PPO (LSTM)
```bash
python training/train_walk_forward.py --config config/training_config_v8_cloud.yaml --recurrent
```

### C. Ensemble (3 modèles - Recommandé)
```bash
python training/train_walk_forward.py --config config/training_config_v8_cloud.yaml --ensemble 3
```

### D. Optimisation Hyperparamètres (Optuna)
```bash
# 4 trials en parallèle sur le GPU (recommandé)
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --n-jobs 4
```

### E. Tests de Robustesse
```bash
python scripts/robustness_tests.py --model models/v8/model.zip --all
```

> **Tip :** Utiliser `screen` ou `nohup` pour garder le process actif après déconnexion SSH.

---

## 5. Récupérer les Résultats

Une fois terminé, rapatriez les modèles (`.zip`) et les logs sur votre PC local.

**Sur votre PC Local (PowerShell) :**

```powershell
# Créer dossier local
mkdir D:\Dev\Github\project_ploutos\runpod_results

# Télécharger (Remplacer IP et PORT)
scp -P PORT -r -i "C:\Users\thoma\.ssh\id_ed25519" root@IP:/workspace/project_ploutos/models D:\Dev\Github\project_ploutos\runpod_results\
scp -P PORT -r -i "C:\Users\thoma\.ssh\id_ed25519" root@IP:/workspace/project_ploutos/logs D:\Dev\Github\project_ploutos\runpod_results\
```

---

## 6. Arrêter l'Instance (Important !)

1.  Vérifiez que vous avez bien reçu les fichiers.
2.  Sur RunPod, cliquez sur **Stop** (Pause facturation GPU) ou **Terminate** (Suppression définitive).
