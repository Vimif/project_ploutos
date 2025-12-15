/**
 * 🎯 PRO ANALYSIS - Module Frontend
 * 
 * Affiche l'analyse technique professionnelle dans le dashboard
 * Utilise l'API /api/pro-analysis/<ticker>
 */

class ProAnalysis {
    constructor() {
        this.currentData = null;
        this.currentTicker = null;
        this.isLoading = false;
        this.cache = {}; // Cache court (2 min)
        this.refreshInterval = null;
        this.initEventListeners();
    }
    
    /**
     * 🔄 Charger l'analyse pour un ticker
     */
    async loadAnalysis(ticker, forceRefresh = false) {
        // ⚠️ ANTI-DOUBLON : Si déjà en cours, skip
        if (this.isLoading) {
            console.log(`⏭️ Skip pro-analysis ${ticker} (chargement en cours)`);
            return;
        }
        
        // 📂 Vérifier le cache (valide 2 min seulement)
        if (!forceRefresh && this.cache[ticker] && (Date.now() - this.cache[ticker].timestamp < 120000)) {
            console.log(`💾 Cache pro-analysis ${ticker}`);
            this.currentData = this.cache[ticker].data;
            this.currentTicker = ticker;
            this.renderSummary();
            this.renderDetails();
            return;
        }
        
        try {
            this.isLoading = true;
            console.log(`🎯 ${forceRefresh ? '🔄 Rafraîchissement' : 'Chargement'} analyse pro pour ${ticker}...`);
            
            const response = await fetch(`/api/pro-analysis/${ticker}`);
            
            if (!response.ok) {
                throw new Error(`Erreur HTTP ${response.status}`);
            }
            
            this.currentData = await response.json();
            this.currentTicker = ticker;
            
            // 💾 Mise en cache (2 min)
            this.cache[ticker] = {
                data: this.currentData,
                timestamp: Date.now()
            };
            
            console.log('✅ Analyse pro:', this.currentData.overall_signal, this.currentData.confidence + '%');
            
            this.renderSummary();
            this.renderDetails();
            
            // 🔄 Démarrer l'auto-refresh si pas déjà actif
            this.startAutoRefresh();
            
        } catch (error) {
            console.error('❌ Erreur chargement analyse pro:', error);
            this.renderError(error.message);
        } finally {
            this.isLoading = false;
        }
    }
    
    /**
     * 🔄 Auto-refresh toutes les 2 minutes
     */
    startAutoRefresh() {
        // Arrêter l'ancien interval si existe
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Nouveau refresh toutes les 2 minutes (120 000 ms)
        this.refreshInterval = setInterval(() => {
            if (this.currentTicker && !this.isLoading) {
                console.log(`🔄 Auto-refresh pro-analysis ${this.currentTicker}`);
                this.loadAnalysis(this.currentTicker, true); // forceRefresh = true
            }
        }, 120000); // 2 minutes
        
        console.log('⏰ Auto-refresh activé (toutes les 2 min)');
    }
    
    /**
     * ⏸️ Arrêter l'auto-refresh
     */
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
            console.log('⏸️ Auto-refresh désactivé');
        }
    }
    
    /**
     * 📊 Afficher le résumé (sidebar gauche)
     */
    renderSummary() {
        const container = document.getElementById('pro-summary');
        if (!container || !this.currentData) return;
        
        const data = this.currentData;
        const signalClass = `signal-${data.overall_signal?.replace('_', '-') || 'unknown'}`;
        const riskClass = `risk-${data.risk_level || 'unknown'}`;
        
        // Afficher l'âge du cache
        const cacheAge = this.cache[this.currentTicker] ? 
            Math.floor((Date.now() - this.cache[this.currentTicker].timestamp) / 1000) : 0;
        
        // 🔥 FIX: Protection accès current_price avec optional chaining
        const currentPrice = data.current_price || data.stats?.price || null;
        const priceDisplay = currentPrice ? `${currentPrice.toFixed(2)} $` : 'N/A';
        
        container.innerHTML = `
            <div class="signal-badge ${signalClass} mb-3">
                ${this.getSignalEmoji(data.overall_signal)} ${(data.overall_signal || 'UNKNOWN').replace('_', ' ')}
            </div>
            
            <div class="mb-3">
                <div class="text-gray-400 text-xs mb-1">Confiance</div>
                <div class="gauge">
                    <div class="gauge-fill" style="width: ${data.confidence || 0}%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);"></div>
                </div>
                <div class="text-right text-xs mt-1">${(data.confidence || 0).toFixed(0)}%</div>
            </div>
            
            <div class="mb-3">
                <div class="text-gray-400 text-xs">Risque</div>
                <div class="${riskClass} font-bold text-lg">${data.risk_level || 'UNKNOWN'}</div>
            </div>
            
            <div class="text-xs bg-gray-800 p-2 rounded mt-3">
                <div class="text-gray-400 mb-1">Prix actuel</div>
                <div class="text-xl font-bold">${priceDisplay}</div>
                <div class="text-gray-500 text-xs mt-1">🔄 il y a ${cacheAge}s</div>
            </div>
        `;
    }
    
    /**
     * 📁 Afficher les détails (sidebar droite)
     */
    renderDetails() {
        const container = document.getElementById('pro-details');
        if (!container || !this.currentData) return;
        
        const data = this.currentData;
        
        let html = '';
        
        // 📈 TENDANCE
        if (data.trend) {
            html += this.renderIndicatorCard(
                'Tendance',
                'trend',
                data.trend.direction || 'unknown',
                data.trend.strength || 0,
                data.trend.explanation || 'N/A',
                {
                    'SMA 200': data.trend.price_vs_sma200 === 'above' ? '🟢 Au-dessus' : '🔴 En-dessous',
                    'Golden Cross': data.trend.golden_cross ? '✅ Oui' : '❌ Non',
                    'Support': data.trend.support_level ? `$${data.trend.support_level.toFixed(2)}` : 'N/A',
                    'Résistance': data.trend.resistance_level ? `$${data.trend.resistance_level.toFixed(2)}` : 'N/A'
                }
            );
        }
        
        // ⚡ MOMENTUM
        if (data.momentum) {
            html += this.renderIndicatorCard(
                'Momentum (RSI)',
                'momentum',
                data.momentum.signal || 'unknown',
                data.momentum.rsi_value || 0,
                data.momentum.explanation || 'N/A',
                {
                    'RSI': `${(data.momentum.rsi_value || 0).toFixed(1)}`,
                    'Zone': this.getRSIZoneEmoji(data.momentum.zone) + ' ' + (data.momentum.zone || 'unknown'),
                    'Divergence': data.momentum.divergence_detected ? `⚠️ ${data.momentum.divergence_type}` : '✅ Non'
                }
            );
        }
        
        // 📉 MACD
        if (data.macd) {
            html += this.renderIndicatorCard(
                'MACD',
                'macd',
                data.macd.signal || 'unknown',
                data.macd.macd_value || 0,
                data.macd.explanation || 'N/A',
                {
                    'MACD': (data.macd.macd_value || 0).toFixed(3),
                    'Signal': (data.macd.signal_value || 0).toFixed(3),
                    'Histogram': (data.macd.histogram_value || 0).toFixed(3),
                    'Croisement': data.macd.crossover ? `🔀 ${data.macd.crossover}` : '➖ Aucun',
                    'Direction': this.getDirectionEmoji(data.macd.histogram_direction)
                }
            );
        }
        
        // 🌊 VOLATILITÉ
        if (data.volatility) {
            html += this.renderIndicatorCard(
                'Volatilité (Bollinger)',
                'volatility',
                data.volatility.price_position || 'unknown',
                data.volatility.bb_width || 0,
                data.volatility.explanation || 'N/A',
                {
                    'BB Upper': data.volatility.bb_upper ? `$${data.volatility.bb_upper.toFixed(2)}` : 'N/A',
                    'BB Middle': data.volatility.bb_middle ? `$${data.volatility.bb_middle.toFixed(2)}` : 'N/A',
                    'BB Lower': data.volatility.bb_lower ? `$${data.volatility.bb_lower.toFixed(2)}` : 'N/A',
                    'Squeeze': data.volatility.squeeze_detected ? '💥 OUI !' : '❌ Non'
                }
            );
        }
        
        // 📦 VOLUME
        if (data.volume) {
            html += this.renderIndicatorCard(
                'Volume (OBV)',
                'volume',
                data.volume.obv_trend || 'unknown',
                0,
                data.volume.explanation || 'N/A',
                {
                    'Tendance OBV': this.getDirectionEmoji(data.volume.obv_trend),
                    'Confirmation': data.volume.volume_confirmation ? '✅ Oui' : '⚠️ Non',
                    'Smart Money': data.volume.smart_money_accumulation ? '👑 Détecté' : '❌ Non'
                }
            );
        }
        
        // 📄 PLAN DE TRADING
        if (data.trading_plan) {
            html += `
                <div class="bg-gray-800 p-3 rounded-lg mt-3">
                    <div class="font-bold text-sm mb-2">📌 Plan de Trading</div>
                    <div class="text-xs whitespace-pre-wrap">${data.trading_plan}</div>
                </div>
            `;
        }
        
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
     * 🧹 Nettoyer le cache (appel manuel si besoin)
     */
    clearCache() {
        this.cache = {};
        console.log('🧹 Cache Pro Analysis vidé');
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
        if (!direction) return '➡️';
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
                // Délai de 500ms pour éviter les appels simultanés
                setTimeout(() => {
                    this.loadAnalysis(event.detail.ticker);
                }, 500);
            }
        });
        
        // Arrêter l'auto-refresh quand l'onglet est inactif (optimisation)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('💤 Onglet inactif - pause auto-refresh');
                this.stopAutoRefresh();
            } else {
                console.log('👁️ Onglet actif - reprise auto-refresh');
                if (this.currentTicker) {
                    this.startAutoRefresh();
                }
            }
        });
    }
}

// 🚀 Auto-init
if (typeof window.ProAnalysis === 'undefined') {
    window.ProAnalysis = new ProAnalysis();
    console.log('✅ ProAnalysis module loaded');
}
