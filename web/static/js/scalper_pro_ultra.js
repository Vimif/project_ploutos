/**
 * ⚡ SCALPER PRO ULTRA - Core JavaScript
 * Gestion temps réel, signaux ML, heatmap, analytics
 */

class ScalperProUltra {
    constructor() {
        this.currentTicker = 'AAPL';
        this.currentTimeframe = '5m';
        this.refreshInterval = 5000;
        this.chart = null;
        this.currentChartData = null;
        
        this.watchlist = [
            'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
            'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN',
            'JPM', 'BAC', 'WFC', 'GS', 'C', 'AXP',
            'V', 'MA', 'PYPL', 'BLOCK',
            'NFLX', 'DIS', 'CMCSA'
        ];
        
        this.signals = [];
        this.marketData = {};
        
        this.init();
    }

    init() {
        console.log('⚡ Initialisation Scalper Pro Ultra...');
        
        this.setupEventListeners();
        this.loadMainChart();
        this.loadHeatmap();
        this.loadSignals();
        this.updateServerTime();
        
        setInterval(() => this.refreshAll(), this.refreshInterval);
        setInterval(() => this.updateServerTime(), 1000);
    }

    setupEventListeners() {
        const tickerInput = document.getElementById('tickerInput');
        tickerInput.addEventListener('change', (e) => {
            this.currentTicker = e.target.value.toUpperCase();
            this.loadMainChart();
        });

        document.querySelectorAll('.timeframe-pill').forEach(pill => {
            pill.addEventListener('click', (e) => {
                document.querySelectorAll('.timeframe-pill').forEach(p => p.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.tf;
                this.loadMainChart();
            });
        });

        document.getElementById('btnQuickBuy').addEventListener('click', () => this.quickTrade('BUY'));
        document.getElementById('btnQuickSell').addEventListener('click', () => this.quickTrade('SELL'));
    }

    async loadMainChart() {
        try {
            const period = this.timeframeToPeriod(this.currentTimeframe);
            const response = await fetch(`/api/chart/${this.currentTicker}?period=${period}`);
            const data = await response.json();

            if (data.success) {
                this.currentChartData = data;
                this.updatePriceStats(data);
                this.renderChart(data);
                this.renderIndicators(data);
            } else {
                console.error('Erreur chargement chart:', data.error);
                this.showChartError(`Erreur: ${data.error}`);
            }
        } catch (error) {
            console.error('Erreur loadMainChart:', error);
            this.showChartError('Impossible de charger les données');
        }
    }

    showChartError(message) {
        const container = document.getElementById('mainChart');
        container.innerHTML = `
            <div class="chart-loading">
                <i class="bi bi-exclamation-triangle" style="font-size: 48px; color: #ff3366; opacity: 0.5;"></i>
                <p style="margin-top: 12px; color: #9ca3af;">${message}</p>
            </div>
        `;
    }

    timeframeToPeriod(tf) {
        // 🔥 TOUS LES TIMEFRAMES UTILISENT 3mo POUR GARANTIR 60+ BARRES
        const mapping = {
            '1m': '3mo',
            '5m': '3mo',
            '15m': '3mo',
            '1h': '3mo',
            '4h': '6mo'
        };
        return mapping[tf] || '3mo';
    }

    updatePriceStats(data) {
        document.getElementById('currentPrice').textContent = `$${data.current_price.toFixed(2)}`;
        
        const changeEl = document.getElementById('priceChange');
        const changePct = data.price_change_pct.toFixed(2);
        changeEl.textContent = `${changePct > 0 ? '+' : ''}${changePct}%`;
        changeEl.className = `stat-value mono ${changePct >= 0 ? 'green' : 'red'}`;
        
        document.getElementById('volume24h').textContent = this.formatVolume(data.volume_24h);
    }

    renderChart(data) {
        const container = document.getElementById('mainChart');
        container.innerHTML = '<canvas id="chartCanvas"></canvas>';
        
        const ctx = document.getElementById('chartCanvas').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Prix',
                        data: data.close,
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: 'SMA 20',
                        data: data.indicators.sma_20,
                        borderColor: '#ffd700',
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(21, 27, 61, 0.95)',
                        titleColor: '#e5e7eb',
                        bodyColor: '#9ca3af',
                        borderColor: '#2d3748',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(45, 55, 72, 0.3)'
                        },
                        ticks: {
                            color: '#9ca3af',
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        display: true,
                        position: 'right',
                        grid: {
                            color: 'rgba(45, 55, 72, 0.3)'
                        },
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    renderIndicators(data) {
        const container = document.getElementById('indicatorsGrid');
        
        const indicators = data.indicators;
        const rsi = this.getLastValue(indicators.rsi);
        const macd = this.getLastValue(indicators.macd);
        const macdSignal = this.getLastValue(indicators.macd_signal);
        const stochK = this.getLastValue(indicators.stoch_k);
        const adx = this.getLastValue(indicators.adx);
        const atr = this.getLastValue(indicators.atr);
        
        const recommendation = this.calculateRecommendation(rsi, macd, macdSignal, stochK, adx, data.price_change_pct);
        
        const rsiSignal = rsi < 30 ? 'buy' : rsi > 70 ? 'sell' : 'neutral';
        const macdSignal_type = macd > macdSignal ? 'buy' : macd < macdSignal ? 'sell' : 'neutral';
        const stochSignal = stochK < 20 ? 'buy' : stochK > 80 ? 'sell' : 'neutral';
        const adxSignal = adx > 25 ? 'buy' : 'neutral';
        
        container.innerHTML = `
            <!-- Recommandation Globale -->
            <div class="recommendation-card" style="grid-column: 1 / -1;">
                <div>
                    <div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">RECOMMANDATION</div>
                    <div class="recommendation-badge ${recommendation.signal.toLowerCase().replace(' ', '-')}">${recommendation.signal}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; color: #9ca3af; margin-bottom: 6px;">Confiance: ${recommendation.confidence.toFixed(0)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${recommendation.confidence}%;"></div>
                    </div>
                </div>
            </div>
            
            <!-- RSI -->
            <div class="indicator-card">
                <div class="indicator-name">RSI (14)</div>
                <div class="indicator-value" style="color: ${this.getIndicatorColor(rsi, 30, 70)};">
                    ${rsi !== null ? rsi.toFixed(1) : '--'}
                </div>
                <span class="indicator-signal ${rsiSignal}">${this.translateSignal(rsiSignal)}</span>
            </div>
            
            <!-- MACD -->
            <div class="indicator-card">
                <div class="indicator-name">MACD</div>
                <div class="indicator-value" style="color: ${macd > 0 ? '#00ff88' : '#ff3366'};">
                    ${macd !== null ? macd.toFixed(2) : '--'}
                </div>
                <span class="indicator-signal ${macdSignal_type}">${this.translateSignal(macdSignal_type)}</span>
            </div>
            
            <!-- Stochastique -->
            <div class="indicator-card">
                <div class="indicator-name">Stochastique</div>
                <div class="indicator-value" style="color: ${this.getIndicatorColor(stochK, 20, 80)};">
                    ${stochK !== null ? stochK.toFixed(1) : '--'}
                </div>
                <span class="indicator-signal ${stochSignal}">${this.translateSignal(stochSignal)}</span>
            </div>
            
            <!-- ADX -->
            <div class="indicator-card">
                <div class="indicator-name">ADX (Trend)</div>
                <div class="indicator-value" style="color: ${adx > 25 ? '#00d4ff' : '#9ca3af'};">
                    ${adx !== null ? adx.toFixed(1) : '--'}
                </div>
                <span class="indicator-signal ${adxSignal}">${adx > 25 ? 'Trend Fort' : 'Trend Faible'}</span>
            </div>
            
            <!-- ATR -->
            <div class="indicator-card">
                <div class="indicator-name">ATR (Volatilité)</div>
                <div class="indicator-value" style="color: #ffd700;">
                    ${atr !== null ? atr.toFixed(2) : '--'}
                </div>
                <span class="indicator-signal neutral">Volatility</span>
            </div>
            
            <!-- Volume -->
            <div class="indicator-card">
                <div class="indicator-name">Volume 24h</div>
                <div class="indicator-value" style="color: #00d4ff; font-size: 16px;">
                    ${this.formatVolume(data.volume_24h)}
                </div>
                <span class="indicator-signal neutral">Vol</span>
            </div>
        `;
    }

    calculateRecommendation(rsi, macd, macdSignal, stochK, adx, priceChangePct) {
        let score = 50;
        
        if (rsi !== null) {
            if (rsi < 30) score += 15;
            else if (rsi < 40) score += 8;
            else if (rsi > 70) score -= 15;
            else if (rsi > 60) score -= 8;
        }
        
        if (macd !== null && macdSignal !== null) {
            const macdDiff = macd - macdSignal;
            if (macdDiff > 0) score += 12;
            else if (macdDiff < 0) score -= 12;
        }
        
        if (stochK !== null) {
            if (stochK < 20) score += 10;
            else if (stochK < 30) score += 5;
            else if (stochK > 80) score -= 10;
            else if (stochK > 70) score -= 5;
        }
        
        if (adx !== null && adx > 25) {
            if (score > 50) score += 8;
            else if (score < 50) score -= 8;
        }
        
        if (priceChangePct > 2) score += 8;
        else if (priceChangePct > 0) score += 3;
        else if (priceChangePct < -2) score -= 8;
        else if (priceChangePct < 0) score -= 3;
        
        score = Math.max(0, Math.min(100, score));
        
        let signal;
        if (score >= 65) signal = 'BUY';
        else if (score >= 55) signal = 'WEAK BUY';
        else if (score <= 35) signal = 'SELL';
        else if (score <= 45) signal = 'WEAK SELL';
        else signal = 'HOLD';
        
        return { signal, confidence: score };
    }

    getLastValue(arr) {
        if (!arr || arr.length === 0) return null;
        for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i] !== null && arr[i] !== undefined && !isNaN(arr[i])) {
                return arr[i];
            }
        }
        return null;
    }

    getIndicatorColor(value, lowThreshold, highThreshold) {
        if (value === null) return '#9ca3af';
        if (value < lowThreshold) return '#00ff88';
        if (value > highThreshold) return '#ff3366';
        return '#e5e7eb';
    }

    translateSignal(signal) {
        const translations = {
            'buy': 'ACHAT',
            'sell': 'VENTE',
            'neutral': 'NEUTRE'
        };
        return translations[signal] || signal.toUpperCase();
    }

    async loadHeatmap() {
        try {
            // 🔥 UTILISER 3mo POUR GARANTIR 60+ BARRES
            const promises = this.watchlist.slice(0, 20).map(ticker => 
                fetch(`/api/chart/${ticker}?period=3mo`)
                    .then(r => r.json())
                    .catch(err => ({ success: false, symbol: ticker }))
            );
            
            const results = await Promise.all(promises);
            
            const heatmapData = results
                .filter(r => r.success)
                .map(r => ({
                    symbol: r.symbol,
                    change: r.price_change_pct
                }))
                .sort((a, b) => Math.abs(b.change) - Math.abs(a.change))
                .slice(0, 20);
            
            this.renderHeatmap(heatmapData);
        } catch (error) {
            console.error('Erreur loadHeatmap:', error);
        }
    }

    renderHeatmap(data) {
        const grid = document.getElementById('heatmapGrid');
        
        if (data.length === 0) {
            grid.innerHTML = '<p style="color: #9ca3af; text-align: center; padding: 20px;">Aucune donnée disponible</p>';
            return;
        }
        
        grid.innerHTML = '';
        
        data.forEach(item => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            
            const intensity = Math.min(Math.abs(item.change) / 5, 1);
            const color = item.change >= 0 
                ? `rgba(0, 255, 136, ${intensity})`
                : `rgba(255, 51, 102, ${intensity})`;
            
            cell.style.background = color;
            cell.innerHTML = `
                <div class="heatmap-symbol">${item.symbol}</div>
                <div class="heatmap-change">${item.change >= 0 ? '+' : ''}${item.change.toFixed(1)}%</div>
            `;
            
            cell.addEventListener('click', () => {
                this.currentTicker = item.symbol;
                document.getElementById('tickerInput').value = item.symbol;
                this.loadMainChart();
            });
            
            grid.appendChild(cell);
        });
    }

    async loadSignals() {
        try {
            const topTickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'JPM'];
            
            const promises = topTickers.map(ticker => 
                fetch(`/api/pro-analysis/${ticker}`)
                    .then(r => r.json())
                    .then(data => ({ ...data, _ticker: ticker }))
                    .catch(err => ({ error: true, _ticker: ticker }))
            );
            
            const results = await Promise.all(promises);
            
            this.signals = results
                .filter(r => !r.error)
                .filter(r => r.overall_signal !== 'HOLD')
                .map(r => ({
                    ticker: r._ticker,
                    signal: r.overall_signal.includes('BUY') ? 'BUY' : 'SELL',
                    confidence: r.confidence,
                    price: r.current_price,
                    rsi: r.momentum?.rsi_value || 0,
                    strength: this.calculateStrength(r)
                }))
                .sort((a, b) => b.strength - a.strength);
            
            this.renderSignals();
        } catch (error) {
            console.error('Erreur loadSignals:', error);
        }
    }

    calculateStrength(data) {
        let score = data.confidence || 50;
        if (data.trend?.strength) score += data.trend.strength * 0.3;
        if (data.momentum?.rsi_value) {
            const rsi = data.momentum.rsi_value;
            if (rsi < 30 || rsi > 70) score += 10;
        }
        return Math.min(score, 100);
    }

    renderSignals() {
        const container = document.getElementById('signalsGrid');
        
        if (this.signals.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; color: #9ca3af; padding: 40px 20px; grid-column: 1 / -1;">
                    <i class="bi bi-hourglass-split" style="font-size: 32px; opacity: 0.5;"></i>
                    <p style="margin-top: 12px;">Aucun signal fort détecté</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.signals.map(signal => `
            <div class="signal-card ${signal.signal.toLowerCase()}" onclick="scalperPro.selectTicker('${signal.ticker}')">
                <div class="signal-header">
                    <div class="signal-ticker mono">${signal.ticker}</div>
                    <div class="signal-badge ${signal.signal.toLowerCase()}">${signal.signal}</div>
                </div>
                <div class="signal-info">
                    <div class="signal-metric">
                        <span class="metric-label">Prix:</span>
                        <span class="metric-value">$${signal.price?.toFixed(2) || '0.00'}</span>
                    </div>
                    <div class="signal-metric">
                        <span class="metric-label">RSI:</span>
                        <span class="metric-value">${signal.rsi.toFixed(0)}</span>
                    </div>
                    <div class="signal-metric">
                        <span class="metric-label">Score:</span>
                        <span class="metric-value">${signal.strength.toFixed(0)}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }

    selectTicker(ticker) {
        this.currentTicker = ticker;
        document.getElementById('tickerInput').value = ticker;
        this.loadMainChart();
    }

    updateServerTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('fr-FR');
        document.getElementById('serverTime').textContent = timeStr;
    }

    async quickTrade(action) {
        const confirmation = confirm(`⚠️ Confirmer ${action} ${this.currentTicker} ?`);
        if (!confirmation) return;
        
        alert(`⚡ ${action} order placé pour ${this.currentTicker}\n(Intégration Alpaca en cours...)`);
    }

    refreshAll() {
        this.loadMainChart();
        this.loadHeatmap();
        this.loadSignals();
    }

    formatVolume(vol) {
        if (vol >= 1e9) return (vol / 1e9).toFixed(2) + 'B';
        if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
        if (vol >= 1e3) return (vol / 1e3).toFixed(2) + 'K';
        return vol.toString();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.scalperPro = new ScalperProUltra();
});
