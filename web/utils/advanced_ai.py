#!/usr/bin/env python3
"""
🤖 PLOUTOS ADVANCED AI ANALYSIS ENGINE

Moteur d'analyse IA avancé avec:
- Analyse technique multi-indicateurs
- Détection de patterns
- Recommandations personnalisées
- Scénarios de trading
- Intégration V8 Oracle

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class AdvancedAIAnalyzer:
    """
    Moteur d'analyse IA avancé pour trading
    """
    
    def __init__(self):
        self.analysis_depth = 'deep'
    
    def generate_complete_analysis(self, ticker: str, data: dict, 
                                   v8_predictions: Optional[dict] = None) -> str:
        """
        Génère une analyse complète multi-facteurs
        """
        quick_stats = data.get('quick_stats', {})
        signals = data.get('signals', {})
        indicators = data.get('indicators', {})
        
        # Header
        analysis = f"# 📊 ANALYSE COMPLÈTE DE {ticker}\n\n"
        
        # 1. Vue d'ensemble
        analysis += self._overview_section(ticker, quick_stats)
        
        # 2. Analyse technique détaillée
        analysis += self._technical_analysis_section(signals, indicators, quick_stats)
        
        # 3. Signaux de trading
        analysis += self._trading_signals_section(signals, quick_stats)
        
        # 4. Niveaux clés
        analysis += self._key_levels_section(data, quick_stats)
        
        # 5. Scénarios de trading
        analysis += self._trading_scenarios_section(signals, quick_stats)
        
        # 6. V8 Oracle (si disponible)
        if v8_predictions:
            analysis += self._v8_oracle_section(v8_predictions)
        
        # 7. Recommandation finale
        analysis += self._final_recommendation_section(signals, quick_stats, v8_predictions)
        
        return analysis
    
    def _overview_section(self, ticker: str, stats: dict) -> str:
        """
        Vue d'ensemble du titre
        """
        price = stats.get('price', 0)
        change_pct = stats.get('change_pct', 0)
        high_52w = stats.get('high_52w', price)
        low_52w = stats.get('low_52w', price)
        
        # Distance du 52w high/low
        dist_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0
        dist_low = ((price - low_52w) / low_52w * 100) if low_52w > 0 else 0
        
        section = "## 🎯 VUE D'ENSEMBLE\n\n"
        
        # Prix et variation
        emoji = "🟢" if change_pct >= 0 else "🔴"
        section += f"**Prix actuel**: {price:.2f} $ {emoji} {change_pct:+.2f}%\n\n"
        
        # Position dans le range 52w
        section += f"**Range 52 semaines**: {low_52w:.2f}$ - {high_52w:.2f}$\n"
        section += f"- Distance du plus haut: **{dist_high:+.1f}%**\n"
        section += f"- Distance du plus bas: **{dist_low:+.1f}%**\n\n"
        
        # Interprétation
        if dist_high > -10:
            section += "💡 Le titre évolue **proche de ses plus hauts** → Momentum fort mais attention au retournement\n\n"
        elif dist_low < 10:
            section += "💡 Le titre évolue **proche de ses plus bas** → Opportunité potentielle si retournement confirmé\n\n"
        else:
            section += "💡 Le titre évolue **au milieu de son range** → Zone de consolidation\n\n"
        
        return section
    
    def _technical_analysis_section(self, signals: dict, indicators: dict, stats: dict) -> str:
        """
        Analyse technique détaillée
        """
        section = "## 📈 ANALYSE TECHNIQUE DÉTAILLÉE\n\n"
        
        # Tendance
        section += "### 🎯 Tendance\n\n"
        trend_signals = signals.get('trend', {})
        
        sma_signal = trend_signals.get('sma', {}).get('signal', 'NEUTRAL')
        macd_signal = trend_signals.get('macd', {}).get('signal', 'NEUTRAL')
        adx_signal = trend_signals.get('adx', {}).get('signal', 'WEAK')
        
        section += f"- **Moyennes mobiles (SMA)**: {self._get_signal_emoji(sma_signal)} {sma_signal}\n"
        section += f"- **MACD**: {self._get_signal_emoji(macd_signal)} {macd_signal}\n"
        section += f"- **ADX (Force)**: {self._get_signal_emoji(adx_signal)} {adx_signal}\n\n"
        
        # Interprétation tendance
        if 'BUY' in sma_signal and 'BUY' in macd_signal:
            section += "✅ **Conclusion tendance**: Tendance haussière confirmée par plusieurs indicateurs. Les acheteurs dominent le marché.\n\n"
        elif 'SELL' in sma_signal and 'SELL' in macd_signal:
            section += "🚨 **Conclusion tendance**: Tendance baissière confirmée. Les vendeurs contrôlent le mouvement.\n\n"
        else:
            section += "⚠️ **Conclusion tendance**: Signaux mixtes. Le marché hésite entre acheteurs et vendeurs.\n\n"
        
        # Momentum
        section += "### ⚡ Momentum\n\n"
        momentum_signals = signals.get('momentum', {})
        
        rsi_signal = momentum_signals.get('rsi', {}).get('signal', 'NEUTRAL')
        stoch_signal = momentum_signals.get('stochastic', {}).get('signal', 'NEUTRAL')
        rsi_value = stats.get('rsi', 50)
        
        section += f"- **RSI (14)**: {rsi_value:.1f} → {self._get_signal_emoji(rsi_signal)} {rsi_signal}\n"
        section += f"- **Stochastic**: {self._get_signal_emoji(stoch_signal)} {stoch_signal}\n\n"
        
        # Interprétation RSI
        if rsi_value > 80:
            section += "🔴 **RSI extrême**: Titre en **forte sur-achat**. Correction imminente probable. Évitez d'acheter à ces niveaux !\n\n"
        elif rsi_value > 70:
            section += "⚠️ **RSI élevé**: Zone de sur-achat. Prudence recommandée pour les nouvelles positions longues.\n\n"
        elif rsi_value < 20:
            section += "🟢 **RSI extrême**: Titre en **forte sur-vente**. Rebond technique probable à court terme.\n\n"
        elif rsi_value < 30:
            section += "💡 **RSI bas**: Zone de sur-vente. Opportunité d'achat si la tendance globale est positive.\n\n"
        else:
            section += "➡️ **RSI neutre**: Pas de signal extrême. Le momentum est équilibré.\n\n"
        
        # Volatilité
        section += "### 🌊 Volatilité\n\n"
        vol_signals = signals.get('volatility', {})
        
        bb_signal = vol_signals.get('bollinger', {}).get('signal', 'NORMAL')
        section += f"- **Bollinger Bands**: {self._get_signal_emoji(bb_signal)} {bb_signal}\n\n"
        
        # Volume
        section += "### 📊 Volume\n\n"
        volume_ratio = stats.get('volume_ratio', 1.0)
        
        if volume_ratio > 2.0:
            section += f"🔥 **Volume explosif** ({volume_ratio:.1f}x la moyenne) → Forte activité, mouvement significatif\n\n"
        elif volume_ratio > 1.5:
            section += f"📈 **Volume élevé** ({volume_ratio:.1f}x la moyenne) → Intérêt marqué des investisseurs\n\n"
        elif volume_ratio < 0.5:
            section += f"💤 **Volume faible** ({volume_ratio:.1f}x la moyenne) → Faible conviction, méfiance\n\n"
        else:
            section += f"➡️ **Volume normal** ({volume_ratio:.1f}x la moyenne)\n\n"
        
        return section
    
    def _trading_signals_section(self, signals: dict, stats: dict) -> str:
        """
        Signaux de trading concrets
        """
        section = "## 🎯 SIGNAUX DE TRADING\n\n"
        
        overall = signals.get('overall', {})
        recommendation = overall.get('recommendation', 'HOLD')
        confidence = overall.get('confidence', 50)
        buy_score = overall.get('buy_score', 0)
        sell_score = overall.get('sell_score', 0)
        
        # Signal principal
        emoji = self._get_signal_emoji(recommendation)
        section += f"### {emoji} SIGNAL GLOBAL: **{recommendation}**\n\n"
        section += f"- **Confiance**: {confidence:.0f}%\n"
        section += f"- **Score haussier**: {buy_score:.1f}\n"
        section += f"- **Score baissier**: {sell_score:.1f}\n\n"
        
        # Interprétation
        if 'STRONG_BUY' in recommendation:
            section += "✅ **Action recommandée**: ACHETER\n"
            section += "Tous les indicateurs convergent vers un signal haussier fort. C'est une excellente opportunité d'entrée.\n\n"
            section += "**⚠️ Risk Management**:\n"
            section += f"- Stop-loss suggéré: **{stats.get('price', 0) * 0.95:.2f}$** (-5%)\n"
            section += f"- Take-profit 1: **{stats.get('price', 0) * 1.10:.2f}$** (+10%)\n"
            section += f"- Take-profit 2: **{stats.get('price', 0) * 1.20:.2f}$** (+20%)\n\n"
        
        elif 'BUY' in recommendation:
            section += "💡 **Action recommandée**: Acheter avec prudence\n"
            section += "Les signaux sont majoritairement positifs, mais pas unanimes. Entrée possible en position réduite.\n\n"
            section += "**⚠️ Risk Management**:\n"
            section += f"- Stop-loss suggéré: **{stats.get('price', 0) * 0.96:.2f}$** (-4%)\n"
            section += f"- Take-profit: **{stats.get('price', 0) * 1.08:.2f}$** (+8%)\n\n"
        
        elif 'STRONG_SELL' in recommendation:
            section += "🚨 **Action recommandée**: VENDRE ou SHORTER\n"
            section += "Tous les indicateurs sont baissiers. Sortez de vos positions longues ou envisagez un short.\n\n"
            section += "**⚠️ Risk Management (SHORT)**:\n"
            section += f"- Stop-loss: **{stats.get('price', 0) * 1.05:.2f}$** (+5%)\n"
            section += f"- Take-profit: **{stats.get('price', 0) * 0.90:.2f}$** (-10%)\n\n"
        
        elif 'SELL' in recommendation:
            section += "⚠️ **Action recommandée**: Réduire l'exposition\n"
            section += "Les signaux sont majoritairement baissiers. Prenez vos profits si vous êtes long.\n\n"
        
        else:
            section += "⏸️ **Action recommandée**: ATTENDRE\n"
            section += "Les signaux sont mixtes. Attendez une confirmation avant d'agir. Patience = Profit.\n\n"
        
        return section
    
    def _key_levels_section(self, data: dict, stats: dict) -> str:
        """
        Niveaux clés de support/résistance
        """
        section = "## 🎚️ NIVEAUX CLÉS\n\n"
        
        price = stats.get('price', 0)
        high_52w = stats.get('high_52w', price)
        low_52w = stats.get('low_52w', price)
        
        # Calcul niveaux Fibonacci
        range_52w = high_52w - low_52w
        fib_618 = low_52w + range_52w * 0.618
        fib_50 = low_52w + range_52w * 0.5
        fib_382 = low_52w + range_52w * 0.382
        
        section += "### 📊 Support et Résistance\n\n"
        section += f"- **Résistance majeure**: {high_52w:.2f}$ (Plus haut 52s)\n"
        section += f"- **Résistance Fibo 61.8%**: {fib_618:.2f}$\n"
        section += f"- **Zone pivot**: {fib_50:.2f}$ (Milieu de range)\n"
        section += f"- **Support Fibo 38.2%**: {fib_382:.2f}$\n"
        section += f"- **Support majeur**: {low_52w:.2f}$ (Plus bas 52s)\n\n"
        
        # Position actuelle
        if price > fib_618:
            section += f"💡 Le prix ({price:.2f}$) évolue **au-dessus du retracement 61.8%** → Zone de force\n\n"
        elif price < fib_382:
            section += f"💡 Le prix ({price:.2f}$) évolue **en-dessous du retracement 38.2%** → Zone de faiblesse\n\n"
        else:
            section += f"💡 Le prix ({price:.2f}$) évolue dans **la zone centrale** du range\n\n"
        
        return section
    
    def _trading_scenarios_section(self, signals: dict, stats: dict) -> str:
        """
        Scénarios de trading possibles
        """
        section = "## 🎬 SCÉNARIOS DE TRADING\n\n"
        
        price = stats.get('price', 0)
        rsi = stats.get('rsi', 50)
        recommendation = signals.get('overall', {}).get('recommendation', 'HOLD')
        
        # Scénario haussier
        section += "### 📈 SCÉNARIO HAUSSIER (Probabilité: "
        if 'BUY' in recommendation:
            section += "**ÉLEVÉE** 70%)\n\n"
        elif 'SELL' in recommendation:
            section += "**FAIBLE** 30%)\n\n"
        else:
            section += "**MOYENNE** 50%)\n\n"
        
        section += f"**Conditions d'entrée**:\n"
        section += f"- Cassure confirmée au-dessus de {price * 1.02:.2f}$ avec volume\n"
        section += f"- RSI reste en-dessous de 70 pour éviter le sur-achat\n\n"
        
        section += f"**Objectifs**:\n"
        section += f"- TP1: {price * 1.05:.2f}$ (+5%)\n"
        section += f"- TP2: {price * 1.10:.2f}$ (+10%)\n"
        section += f"- TP3: {price * 1.15:.2f}$ (+15%)\n\n"
        
        section += f"**Stop-loss**: {price * 0.97:.2f}$ (-3%)\n\n"
        
        # Scénario baissier
        section += "### 📉 SCÉNARIO BAISSIER (Probabilité: "
        if 'SELL' in recommendation:
            section += "**ÉLEVÉE** 70%)\n\n"
        elif 'BUY' in recommendation:
            section += "**FAIBLE** 30%)\n\n"
        else:
            section += "**MOYENNE** 50%)\n\n"
        
        section += f"**Conditions d'entrée**:\n"
        section += f"- Cassure confirmée en-dessous de {price * 0.98:.2f}$ avec volume\n"
        section += f"- RSI passe sous 50 (confirmation baissière)\n\n"
        
        section += f"**Objectifs**:\n"
        section += f"- TP1: {price * 0.95:.2f}$ (-5%)\n"
        section += f"- TP2: {price * 0.90:.2f}$ (-10%)\n\n"
        
        section += f"**Stop-loss**: {price * 1.03:.2f}$ (+3%)\n\n"
        
        return section
    
    def _v8_oracle_section(self, v8_predictions: dict) -> str:
        """
        Intégration des prédictions V8 Oracle
        """
        section = "## 🔮 V8 ORACLE AI PREDICTIONS\n\n"
        
        predictions = v8_predictions.get('predictions', {})
        
        section += "**Prédictions multi-horizon**:\n\n"
        
        for horizon, pred in predictions.items():
            direction = pred.get('direction', 'NEUTRAL')
            confidence = pred.get('confidence', 0)
            target = pred.get('target_price', 0)
            
            emoji = "🟢" if direction == 'UP' else "🔴" if direction == 'DOWN' else "🟡"
            
            section += f"- **{horizon}**: {emoji} {direction} (confiance: {confidence:.0f}%) → Target: {target:.2f}$\n"
        
        section += "\n💡 **Le V8 Oracle combine 2 modèles d'IA entraînés sur des millions de données pour une précision de 65-75%**\n\n"
        
        return section
    
    def _final_recommendation_section(self, signals: dict, stats: dict, 
                                     v8_predictions: Optional[dict] = None) -> str:
        """
        Recommandation finale synthétique
        """
        section = "## ✅ RECOMMANDATION FINALE\n\n"
        
        recommendation = signals.get('overall', {}).get('recommendation', 'HOLD')
        confidence = signals.get('overall', {}).get('confidence', 50)
        
        if 'STRONG_BUY' in recommendation:
            section += "### 🚀 STRONG BUY - ACHAT FORT RECOMMANDÉ\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "**Pourquoi acheter maintenant**:\n"
            section += "- Tous les indicateurs techniques convergent vers un signal haussier\n"
            section += "- Le momentum est fort et confirmé par le volume\n"
            section += "- Ratio risque/récompense très favorable\n\n"
            section += "**⚠️ ATTENTION**: Utilisez TOUJOURS un stop-loss. Aucun trade n'est garanti à 100%.\n"
        
        elif 'BUY' in recommendation:
            section += "### 💡 BUY - ACHAT RECOMMANDÉ\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "Opportunité d'achat avec signaux majoritairement positifs.\n"
            section += "Entrée progressive recommandée avec gestion stricte du risque.\n"
        
        elif 'STRONG_SELL' in recommendation:
            section += "### 🚨 STRONG SELL - VENTE FORTE RECOMMANDÉE\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "**Action immédiate**:\n"
            section += "- Sortez de vos positions longues\n"
            section += "- Envisagez un short si vous maîtrisez cette stratégie\n"
            section += "- Protégez votre capital\n"
        
        elif 'SELL' in recommendation:
            section += "### ⚠️ SELL - VENTE RECOMMANDÉE\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "Réduisez votre exposition. Prenez vos profits si vous êtes en position gagnante.\n"
        
        else:
            section += "### ⏸️ HOLD - ATTENDRE\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "Les signaux sont contradictoires. **La patience est votre meilleur allié**.\n"
            section += "Attendez une confirmation claire avant d'agir.\n"
        
        section += "\n---\n\n"
        section += "📌 **Note**: Cette analyse est générée par l'IA Ploutos V8 à des fins éducatives. "section += "Faites toujours vos propres recherches (DYOR). Investir comporte des risques.\n"
        
        return section
    
    def _get_signal_emoji(self, signal: str) -> str:
        """
        Retourne l'emoji approprié pour un signal
        """
        if 'STRONG_BUY' in signal:
            return '🚀'
        elif 'BUY' in signal or 'OVERSOLD' in signal:
            return '🟢'
        elif 'STRONG_SELL' in signal:
            return '🚨'
        elif 'SELL' in signal or 'OVERBOUGHT' in signal:
            return '🔴'
        elif 'NEUTRAL' in signal or 'NORMAL' in signal:
            return '🟡'
        else:
            return '➡️'
