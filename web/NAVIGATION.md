# 🧭Ploutos - Guide de Navigation

## 🎯 **Objectif**
Barre de navigation unifiée et intuitive pour toutes les pages du dashboard Ploutos.

---

## 📌 **Structure de Navigation**

### **Pages Principales**

| Page | URL | Description |
|------|-----|-------------|
| **Dashboard** | `/` | Vue d'ensemble : Portfolio, Positions, Trades, Métriques |
| **Chart Pro** | `/chart` | Analyse technique complète avec indicateurs avancés |
| **Tools** | `/tools` | Outils : Screener, Backtester, Alertes, Corrélation, Portfolio |

### **Modules d'Analyse**

| Module | URL | Description |
|--------|-----|-------------|
| **V8 Oracle** | `/?tab=v8oracle` | Prédictions multi-horizon (1j, 5j) avec ensemble |
| **V7 Legacy** | `/?tab=v7analysis` | Ancien système à 3 experts (compatibilité) |
| **Analyse Technique** | `/chart` | Patterns, MTF, Fibonacci, Support/Résistance |

---

## ✨ **Fonctionnalités de la Barre de Navigation**

### 1️⃣ **Recherche Rapide**
```html
Clic sur la barre de recherche en haut à droite
Taper un ticker (ex: NVDA)
Appuyer sur Entrée
→ Redirige vers /chart?ticker=NVDA avec auto-chargement
```

### 2️⃣ **Menu Watchlists**
```html
Survol "Watchlists" dans la nav
→ Dropdown avec toutes les watchlists disponibles
Clic sur une watchlist
→ Redirige vers /?watchlist=slug
```

**Watchlists disponibles :**
- 🏆 **Top US** : AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META
- 🔥 **Tech Giants** : Mega-caps technologiques
- 🪙 **Crypto Exposure** : MSTR, COIN, RIOT, MARA, HOOD
- 🏛️ **Banks** : JPM, BAC, WFC, C, GS
- 🇵🇷 **CAC 40** : Actions françaises (Total, LVMH, etc.)
- 🇰🇷 **Korea** : Samsung, Hyundai, LG, etc.
- ... (20 listes au total)

### 3️⃣ **Menu Analyses**
```html
Survol "Analyses" dans la nav
→ Dropdown avec 3 options :
  - V8 Oracle (prédictions IA)
  - V7 Legacy (ancien modèle)
  - Analyse Technique (graphiques)
```

### 4️⃣ **Infos Compte en Temps Réel**
```html
Affiché en haut à droite :
- 🟢 Live (statut connexion)
- 🕒 Heure actuelle
- 💰 Valeur du portfolio (ex: $102,345)
```

### 5️⃣ **Mode Mobile**
```html
Sur mobile/tablette :
- Menu hamburger (☰)
- Menu complet en dropdown vertical
- Recherche rapide accessible
```

---

## 🛠️ **Intégration dans vos Templates**

### **Méthode 1 : Inclusion Flask (Recommandé)**

```python
# Dans app.py
@app.route('/')
def index():
    return render_template('index.html')
```

```html
<!-- Dans index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ploutos Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-gray-900 text-gray-100">
    
    <!-- Include Navigation -->
    {% include 'components/nav.html' %}
    
    <!-- Votre contenu -->
    <main class="container mx-auto px-4 py-6">
        <h1>Bienvenue sur Ploutos</h1>
    </main>
    
</body>
</html>
```

### **Méthode 2 : Base Template (Pour cohérence)**

```html
<!-- web/templates/base.html -->
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Ploutos Dashboard{% endblock %}</title>
    
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    {% block extra_head %}{% endblock %}
</head>
<body class="bg-gray-900 text-gray-100">
    
    <!-- Navigation unifiée -->
    {% include 'components/nav.html' %}
    
    <!-- Contenu de la page -->
    {% block content %}{% endblock %}
    
    <!-- Footer -->
    <footer class="bg-gray-800 mt-8 py-4">
        <div class="container mx-auto px-4 text-center text-gray-400 text-sm">
            © 2025 Ploutos V8 Oracle - AI Trading System
        </div>
    </footer>
    
    {% block extra_scripts %}{% endblock %}
</body>
</html>
```

```html
<!-- Ensuite dans index.html -->
{% extends 'base.html' %}

{% block title %}Dashboard - Ploutos{% endblock %}

{% block content %}
<main class="container mx-auto px-4 py-6">
    <h1>Dashboard</h1>
    <!-- Votre contenu -->
</main>
{% endblock %}
```

---

## 💡 **Flux de Navigation Utilisateur**

### **Scénario 1 : Analyser une action**
```
1. Arriver sur Dashboard (/)
2. Cliquer sur "Chart Pro" dans la nav
3. Taper "NVDA" dans la recherche rapide + Entrée
4. → /chart?ticker=NVDA se charge avec graphique
5. Voir patterns, indicateurs, MTF
6. Cliquer sur "Dashboard" pour revenir
```

### **Scénario 2 : Utiliser une watchlist**
```
1. Arriver sur Dashboard
2. Survoler "Watchlists" dans la nav
3. Cliquer sur "Crypto Exposure"
4. → Dashboard affiche les cartes MSTR, COIN, RIOT, etc.
5. Cliquer sur "Analyser tout" pour batch V8
6. Cliquer sur "Voir graphique" sur MSTR
7. → /chart?ticker=MSTR se charge automatiquement
```

### **Scénario 3 : Prédiction V8 Oracle**
```
1. Arriver sur Dashboard
2. Cliquer sur "Analyses" > "V8 Oracle"
3. → Onglet V8 Oracle s'active
4. Taper "AAPL" + clic "Prédire"
5. Voir prédictions 1j/5j + recommandation
6. Clic "Batch" pour analyser plusieurs tickers
```

---

## 🎨 **Personnalisation**

### **Modifier les couleurs**
```html
<!-- Dans nav.html, modifier les classes Tailwind -->

<!-- Changer couleur primaire (bleu → violet) -->
from-blue-400 to-cyan-400  →  from-purple-400 to-pink-400

<!-- Changer fond nav -->
bg-gradient-to-r from-gray-800 via-gray-900 to-gray-800
→
bg-gradient-to-r from-blue-900 via-indigo-900 to-purple-900
```

### **Ajouter un lien personnalisé**
```html
<!-- Dans nav.html, section <nav> -->
<a href="/ma-page" class="nav-link px-4 py-2 rounded-lg hover:bg-gray-700 transition flex items-center space-x-2">
    <i class="fas fa-rocket"></i>
    <span>Ma Page</span>
</a>
```

### **Ajouter une watchlist au dropdown**
```python
# Dans web/routes/watchlists.py, ajouter dans WATCHLISTS
{
    'slug': 'ma-liste',
    'name': 'Ma Liste Perso',
    'icon': 'fas fa-star',
    'color': 'yellow',
    'tickers': ['AAPL', 'GOOGL', 'MSFT']
}
```

---

## ✅ **Checklist de Déploiement**

- [x] Créer `web/templates/components/nav.html`
- [ ] Modifier `web/templates/index.html` pour inclure nav
- [ ] Modifier `web/templates/advanced_chart.html` pour inclure nav
- [ ] Modifier `web/templates/tools.html` pour inclure nav
- [ ] Créer `web/templates/base.html` (optionnel)
- [ ] Tester navigation sur desktop
- [ ] Tester navigation sur mobile
- [ ] Tester recherche rapide
- [ ] Tester dropdowns (Analyses, Watchlists)
- [ ] Vérifier highlight page active

---

## 🚀 **Avantages**

✅ **Cohérence** : Même nav partout  
✅ **Intuitivité** : Navigation claire et logique  
✅ **Rapidité** : Recherche rapide + accès watchlists  
✅ **Responsive** : Fonctionne mobile/desktop  
✅ **Live** : Affiche portfolio en temps réel  
✅ **Moderne** : Design gradient + animations  

---

## 📝 **TODO - Améliorations Futures**

- [ ] Ajouter notifications (alertes déclenchées)
- [ ] Historique de recherche (localStorage)
- [ ] Favoris utilisateur (cookies)
- [ ] Thème sombre/clair (toggle)
- [ ] Raccourcis clavier (Ctrl+K pour recherche)
- [ ] Breadcrumbs (fil d'Ariane)
- [ ] Menu utilisateur (paramètres, logout)

---

**Questions ?** Consulte le code dans `web/templates/components/nav.html` ou crée une issue GitHub.
