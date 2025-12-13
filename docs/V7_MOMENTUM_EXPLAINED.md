# 🧠 PLOUTOS V7 - MOMENTUM PREDICTOR EXPLAINED

**Date:** 13 Décembre 2025

---

## 🎯 Le Concept en Simple

### Avant (V6 - Complexe) :
```
État du marché → [Réseau RL complexe] → Action (BUY/SELL/HOLD) → Profit/Loss
                      (étapes infinies)        (Réseau apprend par essais/erreurs)
```

Problèmes:
- Trop de variables à optimiser
- NaN/Inf partout
- Impossible de valider

### Maintenant (V7 - Simple) :
```
Features d'AUJOURD'HUI → [Petit réseau] → Prédiction DEMAIN (UP ou DOWN)
  (30 indicateurs)       (128 → 64 → 32)   (Probabilités: 0-100%)
```

Avantages:
- Simple à comprendre
- Facile à valider (vraie prédiction vs réalité)
- Robuste

---

## 📋 Qu'est-ce que le Script Fait ?

### Flux Global :

```
① LOAD DATA
   └─ Charge CSV avec historique OHLCV (Open, High, Low, Close, Volume)
   └─ Ex: AAPL sur 2 ans = 500 jours

② EXTRACT FEATURES (30 par jour)
   └─ RSI, MACD, Bollinger, Volume momentum, etc.
   └─ Représente l'état technique du marché
   └─ Shape: (500 jours, 30 features)

③ NORMALIZE
   └─ StandardScaler: Ramène chaque feature à moyenne=0, std=1
   └─ (RSI 0-100 vs Volume 0-1M doivent être sur même échelle)

④ SPLIT DATA (80% train, 20% test)
   └─ Train: 400 jours → entraîner le réseau
   └─ Test: 100 jours → valider (données neuves)

⑤ CREATE & TRAIN MODEL
   └─ Réseau simple: Input(30) → Dense(128) → Dense(64) → Dense(32) → Output(2)
   └─ Output(2) = [Prob(DOWN), Prob(UP)]
   └─ Entraînement: 100 époques

⑥ EVALUATE
   └─ Accuracy, Precision, Recall, F1-Score, AUC-ROC
   └─ Confusion Matrix: Vrais positifs vs faux positifs

⑦ SAVE MODEL
   └─ best_model.pth (poids du réseau)
   └─ scaler.pkl (normalisation)
   └─ metadata.json (informations)
```

---

## 📊 Les 30 Features En Détail

### Catégorie 1 : PRICE (6 features)
```python
1. returns              # Variation du prix % (ex: +1.5%)
2. price_sma_20        # Prix moyen sur 20 jours
3. price_sma_50        # Prix moyen sur 50 jours
4. price_position      # Dist entre prix et SMA(20) en %
5. high_low_ratio      # Amplitude intraday = (High-Low) / Close
6. close_open_ratio    # (Close-Open) / Open = force du jour
```
**Utilité:** Représente tendance court/moyen terme

### Catégorie 2 : MOMENTUM (9 features)
```python
7.  rsi_14             # Relative Strength Index (0-100)
                       # >70 = overbought, <30 = oversold
8.  rsi_7              # RSI court-terme
9.  macd               # Moving Average Convergence Divergence
10. macd_signal        # Signal du MACD (moyenne exponentielle)
11. macd_histogram     # MACD - Signal (divergence)
12. momentum_10        # Prix maintenant - Prix il y a 10 jours
13. rate_of_change     # ROC = variation entre hier et aujourd
14. stoch_k           # Stochastic Oscillator (position dans range 14j)
```
**Utilité:** "Cet actif bouge-t-il rapidement ?" → Force du mouvement

### Catégorie 3 : VOLATILITY (6 features)
```python
15. volatility_20      # Écart-type des retours (20 jours)
16. atr                # Average True Range (amplitude moyenne)
17. atr_ratio          # ATR en % du prix
18. bb_position        # Position dans les Bandes de Bollinger (0=bas, 1=haut)
19. bb_width           # Largeur des bandes en %
```
**Utilité:** "Le marché est-il calme ou chaotique ?"

### Catégorie 4 : VOLUME (5 features)
```python
20. volume_sma         # Volume moyen (20 jours)
21. volume_ratio       # Volume actuel / Volume moyen
22. price_volume_trend # Prix momentum * Volume
23. on_balance_volume  # Somme cumulée de volumes signés
24. obv_sma            # OBV lissé (20 jours)
```
**Utilité:** "Est-ce qu'il y a de l'intérêt (volume) derrière ce mouvement ?"

### Catégorie 5 : TREND (4 features)
```python
25. ema_12             # Exponent Moving Average 12 jours
26. ema_26             # Exponent Moving Average 26 jours
27. ema_ratio          # EMA(12) / EMA(26) = "signal de crossover"
28. trend_strength     # Force de la tendance (0-1)
```
**Utilité:** "Y a-t-il une vraie tendance ou c'est du bruit ?"

---

## 🧠 Comment Fonctionne le Réseau ?

### Architecture :
```
INPUT LAYER (30 features)
    ↑
    ↑ Weight matrix (30 x 128)
    ↑
 DENSE LAYER 1 (128 neurones)
    ↑ ReLU activation (force non-linéaire)
    ↑ Dropout 30% (coupe aléatoirement 30% des connexions)
    ↑
 DENSE LAYER 2 (64 neurones)
    ↑ ReLU activation
    ↑ Dropout 30%
    ↑
 DENSE LAYER 3 (32 neurones)
    ↑ ReLU activation
    ↑ Dropout 20%
    ↑
 OUTPUT LAYER (2 neurones: [Prob(DOWN), Prob(UP)])
    ↑ Softmax (norm alise à somme = 100%)
    ↑
 PRED: DOWN si logit[0] > logit[1], sinon UP
```

### Exemple Concret :

**Input (30 features pour AAPL aujourd'hui):**
```
rsi_14: 65.2
macd_histogram: 0.5
volume_ratio: 1.2
bb_position: 0.8
... (26 autres features)
```

**Forward Pass:**
```
[30 values] 
  → (multiply by 30x128 weights) 
  → [128 values] 
  → ReLU → [128 values]
  → (multiply by 128x64 weights)
  → [64 values]
  → ReLU → [64 values]
  → (multiply by 64x32 weights)
  → [32 values]
  → ReLU → [32 values]
  → (multiply by 32x2 weights)
  → [logit_down, logit_up] = [0.3, 1.5]
  → Softmax → [P(down)=0.18, P(up)=0.82]
```

**Decision:**
- P(UP) = 82% > P(DOWN) = 18%
- **PRÉDICTION: PRIX MONTE DEMAIN** ✅

---

## 📛 Entraînement (Apprentissage)

### Boucle de Gradient :

```python
For each epoch (1 à 100):
    For each batch (32-64 samples):
        1. Forward: (features) → (predictions)
        2. Loss: CrossEntropy(predictions, actual_labels)
        3. Backward: Calcule gradients avec chain rule
        4. Update: weights -= learning_rate * gradients
           (petit pas dans direction qui réduit loss)
```

### Exemple :

**Jour 1 (Non entraîné):**
```
Input: Features d'AAPL
Préd: [0.5, 0.5]  (random, 50/50)
Actuel: 1 (PRIX A MONTÉ)
Loss: High (très faux)
```

**Jour 50 (Entraînement):**
```
Input: Mêmes features
Préd: [0.3, 0.7]  (meilleur)
Actuel: 1
Loss: Lower
```

**Jour 100 (Converge):**
```
Input: Mêmes features
Préd: [0.15, 0.85]  (quasi certain)
Actuel: 1
Loss: Very Low ✅
```

Les poids se sont adjustés pour donner les bonnes réponses!

---

## ✅ Validation du Modèle

### Métriques :

```
🎯 ACCURACY = (Correct) / (Total)
    Ex: 55/100 = 55% (beat coin flip = 50%)

🎯 PRECISION = (True UP) / (Predicted UP)
    "Quand je dis UP, combien de fois j'ai raison ?"
    Ex: 30 correct / 40 prédits = 75%

🎯 RECALL = (True UP) / (Actual UP)
    "Sur tous les vrais UP, combien j'en attrape ?"
    Ex: 30 correct / 50 réels = 60%

🎯 F1-SCORE = Balance entre Precision & Recall
    Si Precision=75%, Recall=60% → F1=67%

🎯 AUC-ROC = Aire sous la courbe ROC (0-1)
    0.5 = Aléatoire, 1.0 = Parfait, >0.6 = Bon
```

### Confusion Matrix :
```
                Predicted DOWN    Predicted UP
Actual DOWN          10                5      (Vrai/Faux DOWN)
Actual UP             3               15      (Faux/Vrai UP)

True Positives (TP) = 15 (a prédit UP et c'était juste)
False Positives (FP) = 3  (a prédit UP mais c'était DOWN)
True Negatives (TN) = 10  (a prédit DOWN et c'était juste)
False Negatives (FN) = 5  (a prédit DOWN mais c'était UP)
```

---

## 🚀 Comment Utiliser le Modèle (Après Entraînement)

### 1. Charger le modèle :
```python
import torch
import pickle

model = MomentumClassifier(input_dim=30)
model.load_state_dict(torch.load('models/v7_momentum/best_model.pth'))

with open('models/v7_momentum/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### 2. Faire une prédiction pour un nouveau ticker :
```python
# Features d'AUJOURD'HUI pour MSFT
features_msft = [rsi_14, macd, volume_ratio, ...] # 30 features

# Normaliser
features_scaled = scaler.transform([features_msft])

# Prédire
with torch.no_grad():
    logits = model(torch.FloatTensor(features_scaled))
    probs = torch.softmax(logits, dim=1)[0].numpy()

prob_down, prob_up = probs
confidence = max(prob_down, prob_up) * 100
prediction = "UP" if prob_up > 0.5 else "DOWN"

print(f"MSFT demain: {prediction} (Confiance: {confidence:.1f}%)")
```

### 3. Output pour le website :
```json
{
  "ticker": "MSFT",
  "prediction": "UP",
  "confidence": 78.5,
  "sentiment": "Bullish",
  "recommendation": 4,
  "risk_level": "MEDIUM"
}
```

---

## 🎯 Comparaison V6 vs V7

| Aspect | V6 (RL Trading Bot) | V7 (Momentum Predictor) |
|--------|---------------------|------------------------|
| **Objectif** | Trader autonome | Prédictions + scoring |
| **Type** | Reinforcement Learning | Binary Classification |
| **Complexité** | Très haute (NaN!) | Bas (standard ML) |
| **Time-to-value** | Semaines | Jours |
| **Validation** | Diff icile | Facile (acc vs reality) |
| **Production** | Risqué (argent!) | Sécurisé (infos) |
| **Scalabilité** | 1-2 actifs | 100+ actifs |
| **Revenue** | Nul | API premium |

---

## 🚀 Prochaines Étapes

1. **Lancer l'entraînement:**
   ```bash
   python scripts/train_v7_momentum_model.py \
       --data data/historical_daily.csv \
       --output models/v7_momentum \
       --epochs 100
   ```

2. **Attendre résultats** (~2-5 minutes sur GPU)

3. **Voir métriques** (Accuracy, F1, AUC)

4. **Créer 2 autres modèles** (Mean-Reversion, Trend-Following)

5. **Ensemble Voting** (consensus des 3)

6. **API REST** (FastAPI)

7. **Frontend** (React avec prédictions live)

---

## 👋 Questions Fréquentes

**Q: Pourquoi 30 features ?**
A: C'est un equilibre. Trop peu (5) = modèle trop simple. Trop (100+) = overfit.

**Q: Pourquoi 80/20 split ?**
A: Std en ML. Permet entraînement solide + test juste.

**Q: Pourquoi 2 outputs (DOWN, UP) et pas 3 (DOWN, HOLD, UP) ?**
A: Plus simple. HOLD = confiance faible (proche 50/50).

**Q: Ca va battre le marché ?**
A: Si accuracy > 55%, alors oui (au-dessus du coin flip). Mais market timing = difficile.

**Q: Combien d'entraînement ?**
A: 100 époques max. Early stopping si F1 ne s'améliore pas.

---

**C'est beau, non ? Beaucoup plus simple et pragmatique que V6 ! 🚀**
