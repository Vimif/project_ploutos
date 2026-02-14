# Plan Directeur : L'IA de Trading Ultime (Ploutos V8/V9)

Ce document decrit la "Golden Path" pour construire l'IA de trading la plus performante et robuste possible. Inspire des methodes Quant et HFT modernes.

## Objectifs
- **Performance** : Sharpe Ratio > 2.0 (Risque/Rendement excellent)
- **Fiabilite** : Drawdown Max < 15% (Survie aux crises)
- **Robustesse** : Profit constant sur 5+ annees de test OOS (Out-of-Sample)

---

## Phase 1 : Donnees & Univers (Le Socle)

Une IA ne vaut que ce qu'elle mange.

- [x] **1. Univers Dynamique (Selectif)**
    - Au lieu de trader 500 actions (bruit), selectionner chaque trimestre les **50 actions les plus fortes** (Top Momentum + Volatilite suffisante).
    - *Pourquoi ?* L'IA detecte mieux les signaux sur des actifs qui bougent vraiment.
- [x] **2. Donnees Macroeconomiques (Contexte)**
    - Integrer en entree du reseau :
        - **VIX (Volatilite)** : Pour savoir quand etre defensif.
        - **TNX (Taux 10 ans)** : Impacte fortement la Tech.
        - **DXY (Dollar Index)** : Impacte les matieres premieres.
- [x] **3. Profondeur Historique**
    - Recuperer des **donnees horaires (1h) depuis 2010** (minimum 2 cycles economiques : Bull run, Crash Covid, Hausse des taux).

## Phase 2 : Architecture & Modele (Le Cerveau)

- [x] **4. Memoire (LSTM / RecurrentPPO)**
    - Utiliser `RecurrentPPO` (de stable-baselines3-contrib) au lieu de PPO standard.
    - *Avantage* : L'IA se "souvient" des bougies precedentes et du contexte (ex: "ca baisse depuis 3 jours") au lieu de juste voir l'instant T.
- [x] **5. Ensemble Learning (Le Conseil des Sages)**
    - Entrainer **3 a 5 modeles** identiques avec des "seeds" differentes.
    - Pour prendre une decision : Vote a la majorite.
    - *Avantage* : Lisse les erreurs individuelles et augmente considerablement la fiabilite.

## Phase 3 : Protocole d'Entrainement (L'Ecole)

C'est ici que se joue 80% de la performance future.

- [x] **6. Walk-Forward Analysis (Le Gold Standard)**
    - Ne jamais entrainer sur 2010-2020 et tester sur 2021.
    - Faire :
        - Train 2010-2015 -> Test 2016
        - Train 2010-2016 -> Test 2017
        - ...
        - Train 2010-2023 -> Test 2024
    - *Resultat* : Une courbe de performance realiste qui simule le trading reel annee apres annee.
- [ ] **7. Hyperparameter Tuning (Optuna)**
    - Utiliser un script d'optimisation (Optuna) pour trouver le meilleur `learning_rate`, `batch_size`, `gamma` automatiquement. C'est souvent +20% de performance gratuite.

## Phase 4 : Robustesse & Validation (Le Crash Test)

- [x] **8. Monte Carlo Simulations**
    - Lancer 1000 backtests en ajoutant du bruit aleatoire aux prix (+/- 0.1%).
    - Si l'IA perd de l'argent dans >5% des cas, elle est **sur-optimisee** (overfitting) -> Poubelle.
- [x] **9. Stress Test "Krach"**
    - Simuler manuellement une chute de -20% en une journee. Verifier que l'IA coupe ses positions (Stop Loss) ou se met short immediatement.

## Phase 5 : Production (Le Reel)

- [ ] **10. Paper Trading "Smart Check"**
    - Script qui tourne 24/7 sur un VPS (serveur).
    - Verifie les positions toutes les **5-15 minutes** (Stop Loss d'urgence).
    - Prend des decisions de trend toutes les **1h** (Bougies closes).
- [ ] **11. Monitoring Temps Reel**
    - Alertes Discord/Telegram a chaque trade.
    - Dashboard Grafana pour suivre le P&L et l'exposition.

---

## Todo List Immediate (V7 -> V8)

1. [x] Coder `core/macro_data.py` pour recuperer VIX/TNX.
2. [x] Creer l'environnement `UniversalTradingEnvV8_LSTM` (compatible memoire).
3. [x] Mettre en place le script `train_walk_forward.py`.
4. [x] Tester l'approche "Ensemble" sur le S&P 500 actuel.
