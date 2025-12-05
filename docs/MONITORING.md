# 🔍 Guide de Monitoring Production

## 🎯 Vue d'ensemble

Ce guide explique comment utiliser le système de monitoring de Ploutos pour détecter la **dérive de modèle** (model drift) en production.

---

## 📚 Types de Dérive

### **1. Data Drift (Dérive des Données)**

**Définition** : La distribution des features change (P(X) ≠ P'(X))

**Exemples** :
- Volatilité moyenne passe de 15% à 40% (crise COVID)
- Volume moyen double (nouveaux traders retail)
- Corrélations sectorielles changent

**Détection** : PSI (Population Stability Index) + KS Test

---

### **2. Concept Drift (Dérive du Concept)**

**Définition** : La relation X→Y change (P(Y|X) ≠ P'(Y|X))

**Exemples** :
- RSI>70 ne signifie plus "surachété"
- Breakout patterns ne fonctionnent plus
- Mean-reversion disparaît (trending market)

**Détection** : ADDM (Autoregressive Drift Detection Method)

---

### **3. Model Drift (Dérive du Modèle)**

**Définition** : Performance du modèle se dégrade

**Symptômes** :
- Sharpe Ratio : 1.5 → 0.3
- Max Drawdown : -15% → -35%
- Win Rate : 55% → 45%

**Détection** : Comparaison métriques baseline vs actuelles

---

## 🛠️ Utilisation

### **1. Test Local**

```bash
# Tester le drift detector
python3 core/drift_detector.py

# Output attendu :
# ✅ Drift Detector initialisé
# 🟢 Test 1 : Pas de dérive
# 🔴 Test 2 : Dérive détectée
```

---

### **2. Monitoring Production**

```bash
# Monitoring simple
python3 scripts/monitor_production.py --model models/stage1_final.zip

# Monitoring avec auto-retrain
python3 scripts/monitor_production.py --model models/stage1_final.zip --auto-retrain

# Haute sensibilité
python3 scripts/monitor_production.py --model models/stage1_final.zip --sensitivity high
```

---

### **3. Intégration Cron (Monitoring Automatique)**

```bash
# Éditer crontab
crontab -e

# Ajouter monitoring quotidien à 8h
0 8 * * * cd /root/ploutos/project_ploutos && /root/ai-factory/venv/bin/python3 scripts/monitor_production.py --model models/stage1_final.zip >> logs/monitor.log 2>&1

# Monitoring toutes les 6h
0 */6 * * * cd /root/ploutos/project_ploutos && /root/ai-factory/venv/bin/python3 scripts/monitor_production.py --model models/stage1_final.zip --sensitivity high >> logs/monitor.log 2>&1
```

---

## 📊 Métriques de Détection

### **PSI (Population Stability Index)**

```
PSI = Σ (current% - baseline%) * ln(current%/baseline%)

Interprétation :
- PSI < 0.10  : Pas de dérive ✅
- 0.10-0.25   : Dérive modérée ⚠️
- PSI > 0.25  : Dérive critique ❌
```

---

### **KS Test (Kolmogorov-Smirnov)**

```
H0 : Les 2 distributions sont identiques

Si p-value < 0.05 :
  → Rejet H0 → Dérive détectée
```

---

### **Seuils par Sensibilité**

| Métrique | Low | Medium | High |
|----------|-----|--------|------|
| **PSI** | 0.25 | 0.15 | 0.10 |
| **KS** | 0.20 | 0.15 | 0.10 |
| **Performance** | 0.30 | 0.20 | 0.15 |

---

## 🚨 Interprétation des Résultats

### **Exemple 1 : Pas de Dérive**

```
✅ Aucune dérive détectée
   Le modèle fonctionne correctement

📊 Métriques :
   PSI max    : 0.08
   Sharpe     : 1.48
   Max DD     : -11.5%
```

**Action** : Continuer monitoring normal

---

### **Exemple 2 : Data Drift Modéré**

```
🚨 DÉRIVE DÉTECTÉE
  Type     : DATA
  Sévérité : MEDIUM

  Features dérivées (3) :
    - close_norm (PSI: 0.18)
    - volume_norm (PSI: 0.16)
    - rsi (PSI: 0.14)

📋 Recommandations :
  ⚠️ Data Drift détecté (PSI=0.18)
  Features impactées: close_norm, volume_norm, rsi
```

**Action** :
1. Surveiller performance 7 prochains jours
2. Si dégradation continue, retraîner

---

### **Exemple 3 : Model Drift Critique**

```
🚨 DÉRIVE DÉTECTÉE
  Type     : MODEL
  Sévérité : HIGH

📋 Recommandations :
  📉 Model Drift détecté
  Sharpe: 1.50 → 0.75
```

**Action Immédiate** :
1. ⚠️ Arrêter trading live
2. Lancer retraînement : `python3 scripts/train_curriculum.py --stage 1`
3. Valider nouveau modèle (walk-forward)
4. Déployer après tests

---

## 🔄 Stratégies de Réaction

### **1. Retraînement Manuel**

```bash
# 1. Arrêter bot
systemctl stop ploutos-trader-v2.service

# 2. Retraîner
cd /root/ai-factory/tmp/project_ploutos
source /root/ai-factory/venv/bin/activate
python3 scripts/train_curriculum.py --stage 1

# 3. Valider
python3 scripts/monitor_production.py --model models/stage1_final.zip

# 4. Remplacer modèle
cp models/stage1_final.zip /root/ploutos/project_ploutos/models/

# 5. Redémarrer bot
systemctl start ploutos-trader-v2.service
```

---

### **2. Retraînement Automatique (Futur)**

```bash
# Activer auto-retrain
python3 scripts/monitor_production.py \
  --model models/stage1_final.zip \
  --auto-retrain

# Retraîne automatiquement si dérive medium/high
```

---

### **3. Fallback Model**

```python
# Dans bot/trading_bot.py

if drift_detector.detect_drift()['drift_detected']:
    # Basculer vers modèle conservateur
    model = load_fallback_model('models/conservative.zip')
```

---

## 📊 Visualisation

### **Graphiques Générés**

```
reports/
├── drift_monitoring_latest.json  # Dernier rapport
├── drift_report.json             # Historique complet
└── drift_history.png             # Graphique évolution
```

---

### **Dashboard Grafana (Futur)**

**Métriques à tracker** :
- PSI Score (Time Series)
- Sharpe Ratio (Gauge)
- Max Drawdown (Gauge)
- Drift Events (Counter)
- Features dérivées (Table)

---

## ✅ Checklist Production

- [ ] **Baseline établie** : `data_cache/baseline_stats.csv` existe
- [ ] **Performance baseline enregistrée** : Sharpe, Max DD, Win Rate
- [ ] **Monitoring cron activé** : Au moins 1x/jour
- [ ] **Alertes configurées** : Email/Slack si drift > medium
- [ ] **Procédure retraînement documentée** : Checklist claire
- [ ] **Fallback model prêt** : Modèle conservateur en backup
- [ ] **Tests réguliers** : Lancer `monitor_production.py` hebdomadaire
- [ ] **Logs archivés** : `logs/drift_events.jsonl` rotate automatique

---

## 📚 Références

- **PSI** : [Yurdakul (2018) - Statistical Properties of Population Stability Index](https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf)
- **KS Test** : [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- **ADDM** : [Autoregressive Drift Detection Method](https://blog.quantinsti.com/autoregressive-drift-detection-method/)
- **Concept Drift** : [Gama et al. (2014) - A Survey on Concept Drift Adaptation](https://dl.acm.org/doi/10.1145/2523813)

---

## ❓ FAQ

### **Q: À quelle fréquence monitorer ?**
**R** : Dépend du marché
- **Marchés volatils** (crypto) : Toutes les 6h
- **Actions US** : 1x/jour
- **Forex** : 2x/semaine

### **Q: Que faire si dérive persistante ?**
**R** : 3 options
1. Retraîner modèle sur données récentes
2. Changer stratégie (ex: mean-reversion → momentum)
3. Arrêter trading jusqu'à stabilisation marché

### **Q: PSI élevé mais performance OK ?**
**R** : Possible si :
- Modèle robuste aux changements
- Nouveaux patterns bénéfiques

Action : Surveiller, pas d'urgence

### **Q: Comment établir baseline initiale ?**
**R** : Utiliser données train/validation de l'entraînement

```bash
# Sauvegarder baseline après entraînement
cp data_cache/SPY.csv data_cache/baseline_stats.csv
```

---

**Dernière mise à jour** : 5 décembre 2025
