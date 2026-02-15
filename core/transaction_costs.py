#!/usr/bin/env python3
"""
Modèle Avancé de Coûts de Transaction
Simule slippage, impact de marché, et latence
"""

import numpy as np
import pandas as pd


class AdvancedTransactionModel:
    """
    Modèle réaliste de coûts de transaction pour trading algorithmique

    Composantes :
    1. Commission fixe (courtier)
    2. Slippage dynamique (volatilité-dépendant)
    3. Impact de marché (taille ordre vs volume)
    4. Latence (délai exécution)

    Example:
        model = AdvancedTransactionModel()
        exec_price, costs = model.calculate_execution_price(
            ticker='AAPL',
            intended_price=150.0,
            order_size=1000,
            current_volume=5000000,
            side='buy'
        )
    """

    def __init__(
        self,
        base_commission=0.001,  # 0.1% commission de base
        min_slippage=0.0005,  # 0.05% slippage minimum
        max_slippage=0.005,  # 0.5% slippage maximum
        market_impact_coef=0.00015,
        latency_std=0.0002,  # 0.02% latence aléatoire
        vol_ceiling=0.05,  # Max volatility for normalization
        rng=None,
    ):  # RandomState pour reproductibilité
        """
        Args:
            base_commission: Commission fixe du courtier
            min_slippage: Slippage minimum (marchés liquides)
            max_slippage: Slippage maximum (marchés illiquides)
            market_impact_coef: Coefficient d'impact de marché
            latency_std: Écart-type latence (mouvement prix pendant exécution)
            vol_ceiling: Plafond de volatilité pour normalisation du slippage
            rng: np.random.RandomState optionnel (pour reproductibilité)
        """
        self.base_commission = base_commission
        self.min_slippage = min_slippage
        self.max_slippage = max_slippage
        self.market_impact_coef = market_impact_coef
        self.latency_std = latency_std
        self.vol_ceiling = vol_ceiling
        self._rng = rng if rng is not None else np.random

        # Cache pour volatilités (optimisation)
        self.volatility_cache = {}

    def calculate_execution_price(
        self,
        ticker: str,
        intended_price: float,
        order_size: float,
        current_volume: float,
        side: str = "buy",
        recent_prices: pd.Series = None,
    ) -> tuple[float, dict]:
        """
        Calcule le prix d'exécution réel tenant compte de tous les coûts

        Args:
            ticker: Symbol (ex: 'NVDA')
            intended_price: Prix souhaité (limit order)
            order_size: Nombre d'actions
            current_volume: Volume actuel (pour impact de marché)
            side: 'buy' ou 'sell'
            recent_prices: Série de prix récents (pour volatilité)

        Returns:
            (execution_price, costs_breakdown)
        """

        # 1. Slippage basé sur volatilité
        slippage = self._calculate_slippage(ticker, recent_prices)

        # 2. Impact de marché (gros ordres)
        market_impact = self._calculate_market_impact(order_size, current_volume)

        # 3. Latence (mouvement prix pendant exécution)
        latency_cost = self._calculate_latency_cost()

        # 4. Total coûts
        total_cost = self.base_commission + slippage + market_impact + latency_cost

        # 5. Direction dépend du sens
        # Buy = payer plus cher, Sell = recevoir moins
        if side == "buy":
            execution_price = intended_price * (1 + total_cost)
        else:  # sell
            execution_price = intended_price * (1 - total_cost)

        costs_breakdown = {
            "commission": self.base_commission,
            "slippage": slippage,
            "market_impact": market_impact,
            "latency": latency_cost,
            "total_cost": total_cost,
            "total_cost_dollars": abs(order_size * intended_price * total_cost),
        }

        return execution_price, costs_breakdown

    def _calculate_slippage(self, ticker: str, recent_prices: pd.Series = None) -> float:
        """
        Calcule slippage dynamique basé sur volatilité récente

        Principe : Marchés volatils = slippage plus élevé
        """

        if recent_prices is None or len(recent_prices) < 20:
            # Valeur par défaut si pas de données
            return (self.min_slippage + self.max_slippage) / 2

        # Calculer volatilité récente (20 périodes)
        returns = recent_prices.pct_change().dropna()
        volatility = returns.std()

        # Normaliser volatilité (0-1)
        # Volatilité typique : 0.01-0.05 pour actions
        normalized_vol = np.clip(volatility / self.vol_ceiling, 0, 1)

        # Slippage proportionnel à volatilité
        slippage = self.min_slippage + (self.max_slippage - self.min_slippage) * normalized_vol

        # Cache
        self.volatility_cache[ticker] = volatility

        return slippage

    def _calculate_market_impact(self, order_size: float, current_volume: float) -> float:
        """
        Calcule l'impact de l'ordre sur le marché

        Principe : Gros ordres par rapport au volume = impact plus fort

        Modèle simplifié : impact = coef * sqrt(order_size / volume)
        (Modèle réel : Almgren-Chriss, mais trop complexe)
        """

        if current_volume <= 0:
            # Marché illiquide = impact maximum
            return self.max_slippage

        # Ratio ordre/volume
        volume_ratio = order_size / current_volume

        # Impact non-linéaire (racine carrée)
        # Gros ordres ont impact disproportionné
        impact = self.market_impact_coef * np.sqrt(volume_ratio)

        # Clipper pour éviter valeurs absurdes
        impact = np.clip(impact, 0, self.max_slippage)

        return impact

    def _calculate_latency_cost(self) -> float:
        """
        Simule le coût de latence (mouvement prix pendant exécution)

        En production :
        - Latence réseau : 5-50ms
        - Latence bourse : 10-100ms
        - Prix peut bouger pendant ce temps

        Simulation : Bruit aléatoire gaussien
        """

        # Bruit aléatoire (peut être positif ou négatif)
        latency = self._rng.normal(0, self.latency_std)

        # Retourner valeur absolue (coût toujours positif)
        return abs(latency)

    def estimate_total_cost(
        self,
        ticker: str,
        price: float,
        order_size: float,
        volume: float,
        side: str = "buy",
        recent_prices: pd.Series = None,
    ) -> dict:
        """
        Estime le coût total d'un trade AVANT exécution

        Utile pour :
        - Position sizing
        - Validation ordre
        - Optimisation stratégie

        Returns:
            Dict avec estimation coûts en $ et %
        """

        exec_price, costs = self.calculate_execution_price(
            ticker=ticker,
            intended_price=price,
            order_size=order_size,
            current_volume=volume,
            side=side,
            recent_prices=recent_prices,
        )

        notional_value = price * order_size
        total_cost_dollars = costs["total_cost_dollars"]
        total_cost_pct = costs["total_cost"] * 100

        price_diff = exec_price - price
        price_diff_pct = (price_diff / price) * 100

        return {
            "intended_price": price,
            "execution_price": exec_price,
            "price_difference": price_diff,
            "price_difference_pct": price_diff_pct,
            "notional_value": notional_value,
            "total_cost_dollars": total_cost_dollars,
            "total_cost_pct": total_cost_pct,
            "breakdown": costs,
            "is_acceptable": total_cost_pct < 0.5,  # Seuil acceptable : < 0.5%
        }


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    """
    Démonstration du modèle de coûts
    """

    print("\n" + "=" * 80)
    print("💰 MODÈLE DE COÛTS DE TRANSACTION")
    print("=" * 80 + "\n")

    # Créer modèle
    model = AdvancedTransactionModel()

    # Simuler prix récents (pour volatilité)
    recent_prices = pd.Series(100 + np.cumsum(np.random.randn(30) * 2))

    # Test 1 : Petit ordre (liquide)
    print("🟢 Test 1 : Petit ordre sur marché liquide (AAPL)")
    print("-" * 60)

    estimate1 = model.estimate_total_cost(
        ticker="AAPL",
        price=150.0,
        order_size=100,  # 100 actions
        volume=50_000_000,  # 50M volume quotidien
        side="buy",
        recent_prices=recent_prices,
    )

    print(f"  Prix souhaité       : ${estimate1['intended_price']:.2f}")
    print(f"  Prix exécution     : ${estimate1['execution_price']:.2f}")
    print(
        f"  Différence         : ${estimate1['price_difference']:.4f} ({estimate1['price_difference_pct']:.3f}%)"
    )
    print(f"  Valeur notionnelle : ${estimate1['notional_value']:,.2f}")
    print(
        f"  Coût total        : ${estimate1['total_cost_dollars']:.2f} ({estimate1['total_cost_pct']:.3f}%)"
    )
    acceptable1 = "\u2705 OUI" if estimate1["is_acceptable"] else "\u274c NON"
    print(f"  Acceptable         : {acceptable1}")

    # Test 2 : Gros ordre (impact marché)
    print("\n🔴 Test 2 : Gros ordre avec impact marché (NVDA)")
    print("-" * 60)

    estimate2 = model.estimate_total_cost(
        ticker="NVDA",
        price=500.0,
        order_size=10_000,  # 10k actions
        volume=5_000_000,  # 5M volume (20% du volume !)
        side="buy",
        recent_prices=recent_prices * 5,  # Plus volatil
    )

    print(f"  Prix souhaité       : ${estimate2['intended_price']:.2f}")
    print(f"  Prix exécution     : ${estimate2['execution_price']:.2f}")
    print(
        f"  Différence         : ${estimate2['price_difference']:.4f} ({estimate2['price_difference_pct']:.3f}%)"
    )
    print(f"  Valeur notionnelle : ${estimate2['notional_value']:,.2f}")
    print(
        f"  Coût total        : ${estimate2['total_cost_dollars']:,.2f} ({estimate2['total_cost_pct']:.3f}%)"
    )
    acceptable2 = "\u2705 OUI" if estimate2["is_acceptable"] else "\u274c NON"
    print(f"  Acceptable         : {acceptable2}")

    # Breakdown détaillé
    print("\n📊 Breakdown Test 2 :")
    for key, value in estimate2["breakdown"].items():
        if key != "total_cost_dollars":
            print(f"    {key:20s}: {value * 100:.4f}%")

    print("\n" + "=" * 80)
    print("✅ Modèle de coûts prêt pour intégration dans UniversalTradingEnv")
    print("=" * 80 + "\n")
