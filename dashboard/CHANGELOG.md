# Changelog - Dashboard Ploutos

## Version 2.0 - 2025-12-09

### 🎉 Nouveautés

#### Métriques Financières Avancées
- **Sharpe Ratio** : Mesure du rendement ajusté au risque (annualisé)
- **Sortino Ratio** : Comme Sharpe mais pénalise uniquement la volatilité baissière
- **Calmar Ratio** : Rapport rendement annualisé / max drawdown
- **Max Drawdown** : Baisse maximale du portfolio avec dates début/fin
- **Profit Factor** : Ratio gains moyens / pertes moyennes
- **Win Rate avancé** : Analyse des paires BUY->SELL rentables

#### Analytics par Symbole
- Statistiques détaillées par ticker
- Historique des trades filtrés
- Volume et prix moyens

#### Architecture Améliorée
- **Connexion PostgreSQL native** avec fallback automatique sur JSON
- **Module analytics.py** dédié aux calculs financiers
- **Pandas/Numpy** pour calculs performants
- **Gestion d'erreurs robuste** avec logging détaillé

### 🔧 Technique

#### Nouveaux Fichiers
```
dashboard/
├── app.py              # Version 2.0 (remplace l'ancien)
├── app_legacy.py       # Ancien dashboard (backup automatique)
├── analytics.py        # Module de calculs financiers
├── requirements_v2.txt # Dépendances v2
└── CHANGELOG.md        # Ce fichier

scripts/
└── patch_dashboard_v2.sh  # Script de déploiement
```

#### Nouveaux Endpoints

**Analytics Avancés**
```bash
# Métriques avancées (Sharpe, Sortino, etc.)
GET /api/analytics/advanced?days=30

# Analytics pour un symbole
GET /api/analytics/symbol/<SYMBOL>?days=30

# Health check système
GET /api/health
```

**Exemples de réponses**
```json
// /api/analytics/advanced
{
  "success": true,
  "data": {
    "performance_ratios": {
      "sharpe_ratio": 1.45,
      "sortino_ratio": 1.82,
      "calmar_ratio": 2.31
    },
    "risk_metrics": {
      "max_drawdown_pct": -8.42,
      "max_drawdown_start": "2025-11-15",
      "max_drawdown_end": "2025-11-22"
    },
    "win_loss": {
      "wins": 45,
      "losses": 23,
      "total_trades": 68,
      "win_rate_pct": 66.18,
      "avg_win": 324.50,
      "avg_loss": 178.23,
      "profit_factor": 1.82
    },
    "by_symbol": {
      "NVDA": {
        "total_trades": 12,
        "buy_count": 6,
        "sell_count": 6,
        "total_volume": 15420.50,
        "avg_price": 485.23
      }
    }
  },
  "source": "postgresql"
}
```

### 📦 Dépendances Ajoutées

```txt
numpy>=1.24.0
pandas>=2.0.0
psycopg2-binary==2.9.9
```

### 🚀 Installation

#### Méthode Automatique (Recommandée)
```bash
cd /root/ploutos/project_ploutos
bash scripts/patch_dashboard_v2.sh
```

Le script :
- ✅ Crée un backup automatique
- ✅ Installe le nouveau dashboard
- ✅ Vérifie PostgreSQL
- ✅ Teste les imports Python
- ✅ Redémarre le service si nécessaire
- ✅ Rollback automatique en cas d'erreur

#### Méthode Manuelle
```bash
# Backup
cp dashboard/app.py dashboard/app_legacy.py

# Installation
cp dashboard/app_v2.py dashboard/app.py

# Dépendances
pip install -r dashboard/requirements_v2.txt

# Test
cd dashboard && python app.py
```

### ⚠️ Compatibilité

#### Compatibilité Ascendante
- ✅ Tous les anciens endpoints fonctionnent
- ✅ Fallback JSON si PostgreSQL indisponible
- ✅ Les templates HTML existants fonctionnent sans modification
- ✅ Configuration inchangée (`.env`, ports, etc.)

#### Différences
- Nouveaux endpoints disponibles (`/api/analytics/*`)
- Logging plus détaillé
- Métriques supplémentaires dans les réponses

### 🔄 Rollback

En cas de problème :
```bash
# Restaurer l'ancien dashboard
cp dashboard/app_legacy.py dashboard/app.py

# Ou restaurer le backup complet
cp -r backups/dashboard_YYYYMMDD_HHMMSS/* dashboard/

# Redémarrer
sudo systemctl restart ploutos-trader-v2.service
```

### 📊 Utilisation des Métriques

#### Sharpe Ratio
- **> 1** : Bon (rendement supérieur au risque)
- **> 2** : Très bon
- **> 3** : Excellent
- **< 0** : Mauvais (mieux vaut le sans risque)

#### Sortino Ratio
- Similaire au Sharpe mais plus réaliste (volatilité haussière OK)
- Généralement plus élevé que Sharpe

#### Calmar Ratio
- Mesure la récupération après pertes
- **> 1** : Bon
- **> 3** : Excellent

#### Max Drawdown
- **< -10%** : Risque modéré
- **< -20%** : Risque élevé
- **< -30%** : Risque très élevé

### 🐛 Bugs Corrigés

#### Version 1.x
- ❌ Pas de métriques risque/rendement
- ❌ Win rate basique (pas de paires BUY->SELL)
- ❌ Pas d'analytics par symbole
- ❌ Mode JSON uniquement

#### Version 2.0
- ✅ Métriques complètes
- ✅ Win rate précis avec analyse des trades
- ✅ Analytics détaillés par symbole
- ✅ PostgreSQL + fallback JSON

### 📝 Notes de Développement

#### Classes Principales

**PortfolioAnalytics** (`dashboard/analytics.py`)
```python
from dashboard.analytics import PortfolioAnalytics

# Créer l'analyseur
analytics = PortfolioAnalytics(trades, daily_summaries)

# Calculer les métriques
metrics = analytics.get_all_metrics()

# Métriques individuelles
sharpe = analytics.sharpe_ratio()
sortino = analytics.sortino_ratio()
max_dd, start, end = analytics.max_drawdown()
```

#### Extensibilité

Ajout facile de nouvelles métriques dans `analytics.py` :
```python
def ma_nouvelle_metrique(self) -> float:
    """Documenter la métrique"""
    # Calculs avec self.df_trades ou self.df_daily
    return result
```

### 🔮 Roadmap

#### Version 2.1 (Prévue)
- [ ] Comparaison avec benchmark SPY
- [ ] Alpha et Beta du portfolio
- [ ] Graphiques temps réel (Chart.js)
- [ ] Export PDF des rapports
- [ ] Alertes sur métriques (Sharpe < 0.5, DD > -15%, etc.)

#### Version 3.0 (Idées)
- [ ] Dashboard React moderne
- [ ] ML pour prédiction drawdowns
- [ ] Backtesting intégré
- [ ] Multi-timeframe analytics

### 🤝 Contribution

Pour contribuer :
1. Créer une branche `feature/ma-feature`
2. Ajouter des tests
3. Documenter dans CHANGELOG.md
4. Pull request

### 📞 Support

En cas de problème :
1. Vérifier les logs : `tail -f logs/dashboard_v2.log`
2. Health check : `curl http://localhost:5000/api/health`
3. Restaurer backup si nécessaire

---

## Version 1.0 - 2025-11-XX

### Fonctionnalités Initiales
- Dashboard Flask basique
- Lecture trades depuis JSON
- Affichage positions Alpaca
- Stats simples (buy/sell count)
- WebSocket pour temps réel

---

**Généré le** : 2025-12-09  
**Auteur** : Vimif  
**Projet** : Ploutos Trading IA  
