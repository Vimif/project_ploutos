# 🧠 Plan Directeur : L'IA de Trading Ultime (Ploutos V8/V9)

Ce document décrit la "Golden Path" pour construire l'IA de trading la plus performante et robuste possible. Inspiré des méthodes Quant et HFT modernes.

## 🎯 Objectifs
- **Performance** : Sharpe Ratio > 2.0 (Risque/Rendement excellent)
- **Fiabilité** : Drawdown Max < 15% (Survie aux crises)
- **Robustesse** : Profit constant sur 5+ années de test OOS (Out-of-Sample)

---

## 🏗️ Phase 1 : Données & Univers (Le Socle)

Une IA ne vaut que ce qu'elle mange.

- [ ] **1. Univers Dynamique (Sélectif)**
    - Au lieu de trader 500 actions (bruit), sélectionner chaque trimestre les **50 actions les plus fortes** (Top Momentum + Volatilité suffisante).
    - *Pourquoi ?* L'IA détecte mieux les signaux sur des actifs qui bougent vraiment.
- [ ] **2. Données Macroéonomiques (Contexte)**
    - Intégrer en entrée du réseau :
        - **VIX (Volatilité)** : Pour savoir quand être défensif.
        - **TNX (Taux 10 ans)** : Impacte fortement la Tech.
        - **DXY (Dollar Index)** : Impacte les matières premières.
- [ ] **3. Profondeur Historique**
    - Récupérer des **données horaires (1h) depuis 2010** (minimum 2 cycles économiques : Bull run, Crash Covid, Hausse des taux).

## 🧠 Phase 2 : Architecture & Modèle (Le Cerveau)

- [ ] **4. Mémoire (LSTM / RecurrentPPO)**
    - Utiliser `RecurrentPPO` (de stable-baselines3-contrib) au lieu de PPO standard.
    - *Avantage* : L'IA se "souvient" des bougies précédentes et du contexte (ex: "ça baisse depuis 3 jours") au lieu de juste voir l'instant T.
- [ ] **5. Ensemble Learning (Le Conseil des Sages)**
    - Entraîner **3 à 5 modèles** identiques avec des "seeds" différentes.
    - Pour prendre une décision : Vote à la majorité.
    - *Avantage* : Lisse les erreurs individuelles et augmente considérablement la fiabilité.

## 🎓 Phase 3 : Protocole d'Entraînement (L'École)

C'est ici que se joue 80% de la performance future.

- [ ] **6. Walk-Forward Analysis (Le Gold Standard)**
    - Ne jamais entraîner sur 2010-2020 et tester sur 2021.
    - Faire :
        - Train 2010-2015 -> Test 2016
        - Train 2010-2016 -> Test 2017
        - ...
        - Train 2010-2023 -> Test 2024
    - *Résultat* : Une courbe de performance réaliste qui simule le trading réel année après année.
- [ ] **7. Hyperparameter Tuning (Optuna)**
    - Utiliser un script d'optimisation (Optuna) pour trouver le meilleur `learning_rate`, `batch_size`, `gamma` automatiquement. C'est souvent +20% de performance gratuite.

## 🛡️ Phase 4 : Robustesse & Validation (Le Crash Test)

- [ ] **8. Monte Carlo Simulations**
    - Lancer 1000 backtests en ajoutant du bruit aléatoire aux prix (+/- 0.1%).
    - Si l'IA perd de l'argent dans >5% des cas, elle est **sur-optimisée** (overfitting) -> Poubelle.
- [ ] **9. Stress Test "Krach"**
    - Simuler manuellement une chute de -20% en une journée. Vérifier que l'IA coupe ses positions (Stop Loss) ou se met short immédiatement.

## 🚀 Phase 5 : Production (Le Réel)

- [ ] **10. Paper Trading "Smart Check"**
    - Script qui tourne 24/7 sur un VPS (serveur).
    - Vérifie les positions toutes les **5-15 minutes** (Stop Loss d'urgence).
    - Prend des décisions de trend toutes les **1h** (Bougies closes).
- [ ] **11. Monitoring Temps Réel**
    - Alertes Discord/Telegram à chaque trade.
    - Dashboard Grafana pour suivre le P&L et l'exposition.

---

## ✅ Todo List Immédiate (V7 -> V8)

1.  [ ] Coder `core/macro_data.py` pour récupérer VIX/TNX.
2.  [ ] Créer l'environnement `UniversalTradingEnvV8_LSTM` (compatible mémoire).
3.  [ ] Mettre en place le script `train_walk_forward.py`.
4.  [ ] Tester l'approche "Ensemble" sur le S&P 500 actuel.
