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
        """Vue d'ensemble du titre"""
        price = stats.get('price', 0)
        change_pct = stats.get('change_pct', 0)
        high_52w = stats.get('high_52w', price)
        low_52w = stats.get('low_52w', price)
        
        dist_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0
        dist_low = ((price - low_52w) / low_52w * 100) if low_52w > 0 else 0
        
        section = "## 🎯 VUE D'ENSEMBLE\n\n"
        emoji = "🟢" if change_pct >= 0 else "🔴"
        section += f"**Prix actuel**: {price:.2f} $ {emoji} {change_pct:+.2f}%\n\n"
        section += f"**Range 52 semaines**: {low_52w:.2f}$ - {high_52w:.2f}$\n"
        section += f"- Distance du plus haut: **{dist_high:+.1f}%**\n"
        section += f"- Distance du plus bas: **{dist_low:+.1f}%**\n\n"
        
        if dist_high > -10:
            section += "💡 Le titre évolue **proche de ses plus hauts** → Momentum fort mais attention au retournement\n\n"
        elif dist_low < 10:
            section += "💡 Le titre évolue **proche de ses plus bas** → Opportunité potentielle si retournement confirmé\n\n"
        else:
            section += "💡 Le titre évolue **au milieu de son range** → Zone de consolidation\n\n"
        
        return section
    
    def _technical_analysis_section(self, signals: dict, indicators: dict, stats: dict) -> str:
        """Analyse technique détaillée"""
        section = "## 📈 ANALYSE TECHNIQUE DÉTAILLÉE\n\n"
        section += "### 🎯 Tendance\n\n"
        
        trend_signals = signals.get('trend', {})
        sma_signal = trend_signals.get('sma', {}).get('signal', 'NEUTRAL')
        macd_signal = trend_signals.get('macd', {}).get('signal', 'NEUTRAL')
        adx_signal = trend_signals.get('adx', {}).get('signal', 'WEAK')
        
        section += f"- **Moyennes mobiles (SMA)**: {self._get_signal_emoji(sma_signal)} {sma_signal}\n"
        section += f"- **MACD**: {self._get_signal_emoji(macd_signal)} {macd_signal}\n"
        section += f"- **ADX (Force)**: {self._get_signal_emoji(adx_signal)} {adx_signal}\n\n"
        
        if 'BUY' in sma_signal and 'BUY' in macd_signal:
            section += "✅ **Conclusion tendance**: Tendance haussière confirmée par plusieurs indicateurs.\n\n"
        elif 'SELL' in sma_signal and 'SELL' in macd_signal:
            section += "🚨 **Conclusion tendance**: Tendance baissière confirmée.\n\n"
        else:
            section += "⚠️ **Conclusion tendance**: Signaux mixtes.\n\n"
        
        section += "### ⚡ Momentum\n\n"
        momentum_signals = signals.get('momentum', {})
        rsi_signal = momentum_signals.get('rsi', {}).get('signal', 'NEUTRAL')
        rsi_value = stats.get('rsi', 50)
        
        section += f"- **RSI (14)**: {rsi_value:.1f} → {self._get_signal_emoji(rsi_signal)} {rsi_signal}\n\n"
        
        if rsi_value > 70:
            section += "⚠️ **RSI élevé**: Zone de sur-achat. Prudence recommandée.\n\n"
        elif rsi_value < 30:
            section += "💡 **RSI bas**: Zone de sur-vente. Opportunité d'achat potentielle.\n\n"
        else:
            section += "➡️ **RSI neutre**: Pas de signal extrême.\n\n"
        
        return section
    
    def _trading_signals_section(self, signals: dict, stats: dict) -> str:
        """Signaux de trading concrets"""
        section = "## 🎯 SIGNAUX DE TRADING\n\n"
        overall = signals.get('overall', {})
        recommendation = overall.get('recommendation', 'HOLD')
        confidence = overall.get('confidence', 50)
        
        emoji = self._get_signal_emoji(recommendation)
        section += f"### {emoji} SIGNAL GLOBAL: **{recommendation}**\n\n"
        section += f"- **Confiance**: {confidence:.0f}%\n\n"
        
        if 'BUY' in recommendation:
            section += "✅ **Action recommandée**: ACHETER\n"
            section += f"**Stop-loss suggéré**: **{stats.get('price', 0) * 0.96:.2f}$** (-4%)\n"
            section += f"**Take-profit**: **{stats.get('price', 0) * 1.08:.2f}$** (+8%)\n\n"
        elif 'SELL' in recommendation:
            section += "⚠️ **Action recommandée**: Réduire l'exposition\n\n"
        else:
            section += "⏸️ **Action recommandée**: ATTENDRE\n\n"
        
        return section
    
    def _key_levels_section(self, data: dict, stats: dict) -> str:
        """Niveaux clés"""
        section = "## 🎚️ NIVEAUX CLÉS\n\n"
        price = stats.get('price', 0)
        high_52w = stats.get('high_52w', price)
        low_52w = stats.get('low_52w', price)
        
        section += f"- **Résistance majeure**: {high_52w:.2f}$\n"
        section += f"- **Support majeur**: {low_52w:.2f}$\n\n"
        
        return section
    
    def _trading_scenarios_section(self, signals: dict, stats: dict) -> str:
        """Scénarios de trading"""
        section = "## 🎬 SCÉNARIOS DE TRADING\n\n"
        price = stats.get('price', 0)
        
        section += f"### 📈 SCÉNARIO HAUSSIER\n\n"
        section += f"**Objectifs**: {price * 1.05:.2f}$ (+5%), {price * 1.10:.2f}$ (+10%)\n"
        section += f"**Stop-loss**: {price * 0.97:.2f}$ (-3%)\n\n"
        
        section += f"### 📉 SCÉNARIO BAISSIER\n\n"
        section += f"**Objectifs**: {price * 0.95:.2f}$ (-5%)\n"
        section += f"**Stop-loss**: {price * 1.03:.2f}$ (+3%)\n\n"
        
        return section
    
    def _v8_oracle_section(self, v8_predictions: dict) -> str:
        """Prédictions V8 Oracle"""
        section = "## 🔮 V8 ORACLE AI PREDICTIONS\n\n"
        predictions = v8_predictions.get('predictions', {})
        
        for horizon, pred in predictions.items():
            direction = pred.get('direction', 'NEUTRAL')
            confidence = pred.get('confidence', 0)
            emoji = "🟢" if direction == 'UP' else "🔴" if direction == 'DOWN' else "🟡"
            section += f"- **{horizon}**: {emoji} {direction} ({confidence:.0f}%)\n"
        
        section += "\n"
        return section
    
    def _final_recommendation_section(self, signals: dict, stats: dict, v8_predictions: Optional[dict] = None) -> str:
        """Recommandation finale"""
        section = "## ✅ RECOMMANDATION FINALE\n\n"
        recommendation = signals.get('overall', {}).get('recommendation', 'HOLD')
        confidence = signals.get('overall', {}).get('confidence', 50)
        
        if 'BUY' in recommendation:
            section += "### 🚀 BUY - ACHAT RECOMMANDÉ\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "⚠️ Utilisez TOUJOURS un stop-loss.\n"
        elif 'SELL' in recommendation:
            section += "### 🚨 SELL - VENTE RECOMMANDÉE\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
        else:
            section += "### ⏸️ HOLD - ATTENDRE\n\n"
            section += f"**Niveau de confiance**: {confidence:.0f}%\n\n"
            section += "La patience est votre meilleur allié.\n"
        
        section += "\n---\n\n"
        section += "📌 **Note**: Cette analyse est générée par l'IA Ploutos V8 à des fins éducatives. "
        section += "Faites toujours vos propres recherches (DYOR). Investir comporte des risques.\n"
        
        return section
    
    def _get_signal_emoji(self, signal: str) -> str:
        """Emoji pour signal"""
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
