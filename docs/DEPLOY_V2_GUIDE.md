# 🚀 GUIDE DÉPLOIEMENT MODÈLE V2 EN PRODUCTION

**Modèle validé** : `ploutos_v2_production.zip` (+139% performance)

**Date** : 7 Décembre 2025

---

## 📍 LOCALISATION DES FICHIERS

### **Sur BBC (Machine d'entraînement)** :

```bash
cd /root/ai-factory/tmp/project_ploutos

# Modèle entraîné
ls -lh models/ploutos_v2_production.zip
ls -lh models/ploutos_v2_production.json

# Checkpoints
ls -lh models/production_v2/checkpoints/
```

### **Sur VPS (Production)** :

```bash
cd /root/ploutos/project_ploutos

# Modèles actuels
ls -lh models/autonomous/
ls -lh models/autonomous/production.zip  # Lien symbolique vers modèle actif
```

---

## 🛠️ FICHIERS QUI CHARGENT LE MODÈLE

### **1. Script Autonome** : `scripts/autonomous_system.py`

**Ligne 698** (fonction `_deploy_model`) :
```python
# Crée automatiquement :
latest_model = "models/autonomous/production.zip"  # ← Modèle utilisé
```

**Ligne 542** (fonction `_validate_model`) :
```python
# Utilise self.model qui a été entraîné dans _train_universal_model()
# Pas besoin de modifier
```

### **2. Bot de Trading** : `scripts/run_trader.py`

**IMPORTANT** : Ce fichier utilise `BrainTrader` qui n'existe plus !

Il faut le mettre à jour pour utiliser le nouveau modèle V2.

---

## 📝 PLAN DE DÉPLOIEMENT

### **ÉTAPE 1 : BACKUP SUR VPS (CRITIQUE)**

```bash
# Sur VPS
cd /root/ploutos/project_ploutos

# Stop le service
sudo systemctl stop ploutos-trader-v2

# Backup complet
sudo cp -r models models_backup_$(date +%Y%m%d_%H%M%S)
sudo cp core/universal_environment.py core/universal_environment_v1_backup.py

echo "✅ Backup créé"
```

---

### **ÉTAPE 2 : COPIER LE NOUVEAU MODÈLE**

```bash
# Depuis BBC vers VPS
scp /root/ai-factory/tmp/project_ploutos/models/ploutos_v2_production.zip \
    root@VPS_IP:/root/ploutos/project_ploutos/models/autonomous/

scp /root/ai-factory/tmp/project_ploutos/models/ploutos_v2_production.json \
    root@VPS_IP:/root/ploutos/project_ploutos/models/autonomous/

# Sur VPS, vérifier
cd /root/ploutos/project_ploutos
ls -lh models/autonomous/ploutos_v2_production.*
```

---

### **ÉTAPE 3 : ACTIVER LE NOUVEL ENVIRONNEMENT**

```bash
# Sur VPS
cd /root/ploutos/project_ploutos

# Remplacer l'environnement
cp core/universal_environment_v2.py core/universal_environment.py

echo "✅ Environnement V2 activé"
```

---

### **ÉTAPE 4 : METTRE À JOUR LE LIEN SYMBOLIQUE**

```bash
# Sur VPS
cd /root/ploutos/project_ploutos/models/autonomous

# Supprimer ancien lien
rm -f production.zip

# Créer nouveau lien vers V2
ln -s ploutos_v2_production.zip production.zip

# Vérifier
ls -lh production.zip
# Doit pointer vers : ploutos_v2_production.zip
```

---

### **ÉTAPE 5 : TESTER EN MODE DRY-RUN**

```bash
# Sur VPS
cd /root/ploutos/project_ploutos

# Test de chargement
python3 << 'EOF'
from stable_baselines3 import PPO
import sys

try:
    model = PPO.load('models/autonomous/production.zip')
    print("✅ Modèle chargé avec succès")
    print(f"Policy: {model.policy}")
    print(f"Device: {model.device}")
    sys.exit(0)
except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
EOF

# Si succès, continuer
```

---

### **ÉTAPE 6 : RELANCER LE SERVICE**

```bash
# Sur VPS
sudo systemctl start ploutos-trader-v2
sudo systemctl status ploutos-trader-v2

# Vérifier les logs
sudo journalctl -u ploutos-trader-v2 -f -n 50
```

**Vérifier que** :
- ✅ Le service démarre sans erreur
- ✅ Le modèle V2 est bien chargé
- ✅ L'environnement V2 est bien utilisé

---

## 🔍 MONITORING (7 PREMIERS JOURS)

### **Dashboard Grafana** : `http://VPS_IP:3000`

**Métriques à surveiller quotidiennement** :

- ✅ **Portfolio Value** : Doit augmenter
- ✅ **Trades/jour** : Doit être > 0 (sinon le bot est bloqué)
- ✅ **Sharpe 7j** : Doit rester > 0.3
- ✅ **Drawdown max** : Doit rester < 10%

### **Logs à vérifier** :

```bash
# Logs temps réel
sudo journalctl -u ploutos-trader-v2 -f

# Logs des 100 dernières lignes
sudo journalctl -u ploutos-trader-v2 -n 100

# Rechercher erreurs
sudo journalctl -u ploutos-trader-v2 | grep -i error
```

---

## ⚠️ ALERTES À CONFIGURER

### **Alertes critiques** :

1. **0 trade pendant 24h**
   - Cause probable : Bot bloqué ou pas de signal
   - Action : Vérifier logs

2. **Drawdown > -5%**
   - Cause probable : Mauvaises décisions
   - Action : Surveiller de près

3. **Sharpe 7j < 0**
   - Cause probable : Performance dégradée
   - Action : Envisager rollback

---

## 🔙 PROCÉDURE DE ROLLBACK

**Si le modèle V2 fait n'importe quoi** :

```bash
# Sur VPS - ROLLBACK IMMÉDIAT
cd /root/ploutos/project_ploutos

# 1. Stopper le service
sudo systemctl stop ploutos-trader-v2

# 2. Restaurer ancien environnement
cp core/universal_environment_v1_backup.py core/universal_environment.py

# 3. Restaurer ancien modèle
cd models/autonomous
rm production.zip
ln -s ANCIEN_MODELE.zip production.zip  # Remplacer par le nom exact

# 4. Relancer
sudo systemctl start ploutos-trader-v2

echo "✅ Rollback terminé"
```

---

## ✅ CHECKLIST DE VALIDATION

### **Semaine 1** :

- [ ] Portfolio > $100k
- [ ] Au moins 5 trades/jour
- [ ] Pas de trade erratique (achat immédiat + vente)
- [ ] Sharpe 7j > 0.3
- [ ] Logs sans erreur

### **Semaine 2-4** :

- [ ] Portfolio > $105k
- [ ] Sharpe 30j > 0.5
- [ ] Drawdown max < 8%
- [ ] Au moins 100 trades total
- [ ] Actions équilibrées (BUY/SELL/HOLD)

### **Après 30 jours** :

- [ ] Portfolio > $110k
- [ ] Performance stable
- [ ] Pas de dérive du comportement
- [ ] **DÉCISION** : Passer en LIVE ou continuer Paper Trading

---

## 📊 RÉSULTATS ATTENDUS

### **Performance Test (30 épisodes)** :

```
💰 PORTFOLIO:
   Moyen : $239,317 (+139.3%)
   Std   : $41,491
   Min   : $176,091
   Max   : $297,658

📈 MÉTRIQUES:
   Sharpe       : 10.000
   Returns Std  : 0.4149

🎯 ACTIONS GLOBALES:
   HOLD  :  27.8%
   BUY   :  38.4%
   SELL  :  33.8%
```

### **En Production (attendu)** :

- 💰 **Portfolio** : +5-15% par mois (conservateur)
- 📈 **Sharpe** : > 1.0 (réaliste)
- 🎯 **Drawdown** : < 10%

**Note** : Les résultats réels seront probablement **moins bons** qu'en backtest (c'est normal).

---

## 📞 CONTACTS D'URGENCE

Si problème critique :

1. **Stopper immédiatement** : `sudo systemctl stop ploutos-trader-v2`
2. **Faire un rollback** (voir procédure ci-dessus)
3. **Analyser les logs** : `sudo journalctl -u ploutos-trader-v2 -n 500 > logs_urgence.txt`
4. **Contacter support** (si applicable)

---

## 📝 NOTES IMPORTANTES

### **⚠️ GARDE EN PAPER TRADING 1-2 SEMAINES MINIMUM**

Ne passe en **LIVE** que si :
- ✅ Performance stable > 30 jours
- ✅ Aucun comportement étrange
- ✅ Sharpe 30j > 1.0
- ✅ Drawdown < 5%

### **🔍 SURVEILLANCE ACTIVE REQUISE**

Pendant les 30 premiers jours :
- Vérifie le dashboard **quotidiennement**
- Analyse les logs **hebdomadairement**
- Compare performance vs backtest

### **📊 PLAN B SI ÉCHEC**

Si performance < +2% après 30 jours :
1. Analyser les trades perdants
2. Ré-entraîner avec plus de données (3-5 ans)
3. Ajuster hyperparams (ent_coef, learning_rate)
4. Envisager curriculum learning

---

## ✅ PRÊT POUR LE DÉPLOIEMENT

Suis les étapes ci-dessus dans l'ordre. 

**Bonne chance ! 🚀**
