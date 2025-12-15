/**
 * 🎯 PRO ANALYSIS - Module Frontend
 * 
 * Affiche l'analyse technique professionnelle dans le dashboard
 * Utilise l'API /api/pro-analysis/<ticker>
 */

class ProAnalysis {
    constructor() {
        this.currentData = null;
        this.initEventListeners();
    }
    
    /**
     * 🔄 Charger l'analyse pour un ticker
     */
    async loadAnalysis(ticker) {
        try {
            console.log(`🎯 Chargement analyse pro pour ${ticker}...`);
            
            const response = await fetch(`/api/pro-analysis/${ticker}`);
            
            if (!response.ok) {
                throw new Error(`Erreur HTTP ${response.status}`);
            }
            
            this.currentData = await response.json();
            console.log('✅ Analyse pro chargée:', this.currentData);
            
            this.renderSummary();
            this.renderDetails();
            
        } catch (error) {
            console.error('❌ Erreur chargement analyse pro:', error);
            this.renderError(error.message);
        }
    }
    
    /**
     * 📊 Afficher le résumé (sidebar gauche)
     */
    renderSummary() {
        const container = document.getElementById('pro-summary');
        if (!container || !this.currentData) return;
        
        const data = this.currentData;
        const signalClass = `signal-${data.overall_signal.replace('_', '-')}`;
        const riskClass = `risk-${data.risk_level}`;
        
        container.innerHTML = `
            <div class="signal-badge ${signalClass} mb-3">
                ${this.getSignalEmoji(data.overall_signal)} ${data.overall_signal.replace('_', ' ')}
            </div>
            
            <div class="mb-3">
                <div class="text-gray-400 text-xs mb-1">Confiance</div>
                <div class="gauge">
                    <div class="gauge-fill" style="width: ${data.confidence}%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);"></div>
                </div>
                <div class="text-right text-xs mt-1">${data.confidence.toFixed(0)}%</div>
            </div>
            
            <div class="mb-3">
                <div class="text-gray-400 text-xs">Risque</div>
                <div class="${riskClass} font-bold text-lg">${data.risk_level}</div>
            </div>
            
            <div class="text-xs bg-gray-800 p-2 rounded mt-3">
                <div class="text-gray-400 mb-1">Prix actuel</div>
                <div class="text-xl font-bold">${data.current_price.toFixed(2)} $</div>
            </div>
        `;
    }
    
    /**
     * 📑 Afficher les détails (sidebar droite)
     */
    renderDetails() {
        const container = document.getElementById('pro-details');
        if (!container || !this.currentData) return;
        
        const data = this.currentData;
        
        let html = '';
        
        // 📈 TENDANCE
        html += this.renderIndicatorCard(
            'Tendance',
            'trend',
            data.trend.direction,
            data.trend.strength,
            data.trend.explanation,
            {
                'SMA 200': data.trend.price_vs_sma200 === 'above' ? '🟢 Au-dessus' : '🔴 En-dessous',
                'Golden Cross': data.trend.golden_cross ? '✅ Oui' : '❌ Non',
                'Support': data.trend.support_level ? `$${data.trend.support_level.toFixed(2)}` : 'N/A',
                'Résistance': data.trend.resistance_level ? `$${data.trend.resistance_level.toFixed(2)}` : 'N/A'
            }
        );
        
        // ⚡ MOMENTUM
        html += this.renderIndicatorCard(
            'Momentum (RSI)',
            'momentum',
            data.momentum.signal,
            data.momentum.rsi_value,
            data.momentum.explanation,
            {
                'RSI': `${data.momentum.rsi_value.toFixed(1)}`,
                'Zone': this.getRSIZoneEmoji(data.momentum.zone) + ' ' + data.momentum.zone,
                'Divergence': data.momentum.divergence_detected ? `⚠️ ${data.momentum.divergence_type}` : '✅ Non'
            }
        );
        
        // 📉 MACD
        html += this.renderIndicatorCard(
            'MACD',
            'macd',
            data.macd.signal,
            data.macd.macd_value,
            data.macd.explanation,
            {
                'MACD': data.macd.macd_value.toFixed(3),
                'Signal': data.macd.signal_value.toFixed(3),
                'Histogram': data.macd.histogram_value.toFixed(3),
                'Croisement': data.macd.crossover ? `🔀 ${data.macd.crossover}` : '➖ Aucun',
                'Direction': this.getDirectionEmoji(data.macd.histogram_direction)
            }
        );
        
        // 🌊 VOLATILITÉ
        html += this.renderIndicatorCard(
            'Volatilité (Bollinger)',
            'volatility',
            data.volatility.price_position,
            data.volatility.bb_width,
            data.volatility.explanation,
            {
                'BB Upper': `$${data.volatility.bb_upper.toFixed(2)}`,
                'BB Middle': `$${data.volatility.bb_middle.toFixed(2)}`,
                'BB Lower': `$${data.volatility.bb_lower.toFixed(2)}`,
                'Squeeze': data.volatility.squeeze_detected ? '💥 OUI !' : '❌ Non'
            }
        );
        
        // 📦 VOLUME
        html += this.renderIndicatorCard(
            'Volume (OBV)',
            'volume',
            data.volume.obv_trend,
            0,
            data.volume.explanation,
            {
                'Tendance OBV': this.getDirectionEmoji(data.volume.obv_trend),
                'Confirmation': data.volume.volume_confirmation ? '✅ Oui' : '⚠️ Non',
                'Smart Money': data.volume.smart_money_accumulation ? '👑 Détecté' : '❌ Non'
            }
        );
        
        // 📄 PLAN DE TRADING
        html += `
            <div class="bg-gray-800 p-3 rounded-lg mt-3">
                <div class="font-bold text-sm mb-2">📌 Plan de Trading</div>
                <div class="text-xs whitespace-pre-wrap">${data.trading_plan}</div>
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    /**
     * 🎴 Créer une carte indicateur
     */
    renderIndicatorCard(title, type, signal, value, explanation, details) {
        let detailsHtml = '';
        for (const [key, val] of Object.entries(details)) {
            detailsHtml += `
                <div class="flex justify-between text-xs mb-1">
                    <span class="text-gray-400">${key}:</span>
                    <span class="font-semibold">${val}</span>
                </div>
            `;
        }
        
        return `
            <div class="indicator-card ${type}">
                <div class="font-bold text-sm mb-2">${title}</div>
                <div class="text-xs text-gray-300 mb-2">${explanation}</div>
                ${detailsHtml}
            </div>
        `;
    }
    
    /**
     * 🚫 Afficher une erreur
     */
    renderError(message) {
        const containers = ['pro-summary', 'pro-details'];
        containers.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.innerHTML = `
                    <div class="text-red-400 text-center py-4">
                        ❌ Erreur<br>
                        <span class="text-xs">${message}</span>
                    </div>
                `;
            }
        });
    }
    
    /**
     * 📦 Helpers
     */
    getSignalEmoji(signal) {
        const emojis = {
            'STRONG_BUY': '🟢🚀',
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴',
            'STRONG_SELL': '🔴⚠️'
        };
        return emojis[signal] || '🔵';
    }
    
    getRSIZoneEmoji(zone) {
        const emojis = {
            'oversold': '🔵',
            'neutral': '⚪',
            'overbought': '🔴'
        };
        return emojis[zone] || '⚪';
    }
    
    getDirectionEmoji(direction) {
        if (direction.includes('increas') || direction === 'rising') return '📈';
        if (direction.includes('decreas') || direction === 'falling') return '📉';
        return '➡️';
    }
    
    /**
     * 🎯 Event Listeners
     */
    initEventListeners() {
        // Auto-reload quand un nouveau ticker est analysé
        window.addEventListener('chartDataLoaded', (event) => {
            if (event.detail && event.detail.ticker) {
                this.loadAnalysis(event.detail.ticker);
            }
        });
    }
}

// 🚀 Auto-init
window.ProAnalysis = new ProAnalysis();
console.log('✅ ProAnalysis module loaded');
