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

- [x] **10. Paper Trading "Smart Check"**
    - Script `scripts/paper_trade.py` opérationnel avec kill switch et monitoring local.
- [ ] **11. Monitoring Temps Reel**
    - Alertes Discord/Telegram a chaque trade.
    - Dashboard Grafana pour suivre le P&L et l'exposition.

---

## Phase 6 : Industrialisation & Intelligence (Ploutos V9)

L'objectif de la V9 est de passer à l'échelle (Scale) et d'ajouter une couche d'intelligence de marché avancée.

- [ ] **12. Architecture Scalable (Performance)**
    - **Shared Memory (Zero-Copy)** : Optimisation critique pour réduire la RAM de 100Go à 5Go lors du training massif.
    - **Polars Data Engine** : Remplacer Pandas par Polars pour un feature engineering 50x plus rapide.
- [ ] **13. Intelligence de Marché (Alpha)**
    - **Détection de Régime (HMM)** : Classifier le marché (Bull/Bear/Volatile) et adapter la stratégie.
    - **Transformer Architecture** : Tester une architecture basée sur l'Attention (Decision Transformer).
- [ ] **14. Qualité Logicielle (CI/CD)**
    - **Tests Unitaires** : Couverture > 80% sur le Core.
    - **Hydra Config** : Gestion modulaire des configurations.

---

## Todo List Immédiate (V8 -> V9)

1. [ ] Mettre en place `pytest` et les premiers tests unitaires.
2. [ ] Implémenter le `SharedMemoryLoader` pour réduire la consommation RAM.
3. [ ] Migrer `hardware.py` vers Hydra ou une classe de config plus robuste.
4. [ ] Refondre le `FeatureEngineer` avec Polars.
