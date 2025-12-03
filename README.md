## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/ploutos_project.git
cd ploutos_project
./scripts/setup_infrastructure.sh
source venv/bin/activate

# Train models
python scripts/train_models.py --sector tech

# Start trading
python scripts/run_trader.py --paper

# View dashboard
streamlit run ui/dashboard.py
```

## 🏗️ Architecture

```
ploutos_project/
├── config/          # Configuration files
├── core/            # Business logic
├── training/        # AI model training
├── trading/         # Trading engine
├── scripts/         # Executable scripts
└── ui/              # Web interfaces
```

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ploutos_project.git
cd ploutos_project
```

2. **Run setup script**
```bash
./scripts/setup_infrastructure.sh
```

3. **Activate environment**
```bash
source venv/bin/activate
```

## 💻 Usage

**Train Models**
```bash
python scripts/train_models.py --sector tech
```

**Paper Trading**
```bash
python scripts/run_trader.py --paper --interval 60
```

**Backtesting**
```bash
python scripts/backtest.py NVDA MSFT --days 180
```

**Dashboard**
```bash
streamlit run ui/dashboard.py
```


### Lancer le trading

Paper trading (simulation)

python scripts/run_trader.py --paper
Live trading

python scripts/run_trader.py
Avec options

python scripts/run_trader.py --capital 50000 --interval 30 --paper


### Dashboard

streamlit run ui/dashboard.py


### Backtesting

python scripts/backtest.py MSFT AAPL NVDA --days 365


## 🧠 Les 4 Cerveaux

- **CRYPTO** (15%) : BTC-USD, ETH-USD, COIN
- **DEFENSIVE** (40%) : SPY, QQQ, VOO
- **ENERGY** (20%) : XOM, CVX, XLE
- **TECH** (25%) : NVDA, MSFT, AAPL, GOOGL

## 🔧 Configuration

Fichiers de config dans `config/`:
- `settings.py` : Paramètres globaux
- `tickers.py` : Organisation des secteurs

## 📝 Logs

Tous les logs dans `/mnt/shared/ploutos_data/logs/` (ou `data/logs/` en local)

## 🤝 Contributing

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Pull Request

## 📜 License

MIT

🎯 UTILISATION COMPLÈTE

# SETUP INITIAL (une seule fois)
cd ~/ploutos_project
./scripts/setup_infrastructure.sh
source venv/bin/activate

# ENTRAÎNER (PC-TOUR avec GPU)
python scripts/train_models.py --sector tech

# LANCER TRADING (PROXMOX 24/7)
python scripts/run_trader.py --paper --interval 60

# DASHBOARD (n'importe où)
streamlit run ui/dashboard.py

# BACKTEST
python scripts/backtest.py NVDA MSFT --days 180
