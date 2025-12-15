/**
 * ⚡ SCALPER PRO ULTRA - Core JavaScript
 * Gestion temps réel, signaux ML, heatmap, analytics
 */

class ScalperProUltra {
    constructor() {
        this.currentTicker = 'AAPL';
        this.currentTimeframe = '5m';
        this.refreshInterval = 5000; // 5 secondes
        this.chart = null;
        this.watchlist = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'INTC',
                          'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ', 'COIN'];
        this.signals = [];
        this.marketData = {};
        
        this.init();
    }

    init() {
        console.log('⚡ Initialisation Scalper Pro Ultra...');
        
        // Event listeners
        this.setupEventListeners();
        
        // Chargement initial
        this.loadMainChart();
        this.loadHeatmap();
        this.loadSignals();
        this.updateServerTime();
        
        // Auto-refresh
        setInterval(() => this.refreshAll(), this.refreshInterval);
        setInterval(() => this.updateServerTime(), 1000);
    }

    setupEventListeners() {
        // Ticker input
        const tickerInput = document.getElementById('tickerInput');
        tickerInput.addEventListener('change', (e) => {
            this.currentTicker = e.target.value.toUpperCase();
            this.loadMainChart();
        });

        // Timeframe pills
        document.querySelectorAll('.timeframe-pill').forEach(pill => {
            pill.addEventListener('click', (e) => {
                document.querySelectorAll('.timeframe-pill').forEach(p => p.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTimeframe = e.target.dataset.tf;
                this.loadMainChart();
            });
        });

        // Quick Buy/Sell
        document.getElementById('btnQuickBuy').addEventListener('click', () => this.quickTrade('BUY'));
        document.getElementById('btnQuickSell').addEventListener('click', () => this.quickTrade('SELL'));
    }

    async loadMainChart() {
        try {
            const period = this.timeframeToPeriod(this.currentTimeframe);
            const response = await fetch(`/api/chart/${this.currentTicker}?period=${period}`);
            const data = await response.json();

            if (data.success) {
                this.updatePriceStats(data);
                this.renderChart(data);
            } else {
                console.error('Erreur chargement chart:', data.error);
            }
        } catch (error) {
            console.error('Erreur loadMainChart:', error);
        }
    }

    timeframeToPeriod(tf) {
        const mapping = {
            '1m': '1d',
            '5m': '5d',
            '15m': '1mo',
            '1h': '3mo',
            '4h': '6mo'
        };
        return mapping[tf] || '5d';
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

    async loadHeatmap() {
        try {
            const promises = this.watchlist.map(ticker => 
                fetch(`/api/chart/${ticker}?period=1d`).then(r => r.json())
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
            // Générer signaux pour top 10 tickers
            const topTickers = this.watchlist.slice(0, 10);
            const promises = topTickers.map(ticker => 
                fetch(`/api/pro-analysis/${ticker}`).then(r => r.json())
            );
            
            const results = await Promise.all(promises);
            
            this.signals = results
                .filter(r => !r.error)
                .filter(r => r.overall_signal !== 'HOLD')
                .map(r => ({
                    ticker: r.current_price ? this.extractTicker(r) : 'UNKNOWN',
                    signal: r.overall_signal.includes('BUY') ? 'BUY' : 'SELL',
                    confidence: r.confidence,
                    price: r.current_price,
                    rsi: r.momentum?.rsi_value || 0,
                    strength: this.calculateStrength(r)
                }))
                .sort((a, b) => b.confidence - a.confidence);
            
            this.renderSignals();
        } catch (error) {
            console.error('Erreur loadSignals:', error);
        }
    }

    extractTicker(data) {
        // Extraire ticker depuis l'objet de réponse
        return this.watchlist.find(t => true) || 'UNKNOWN';
    }

    calculateStrength(data) {
        // Score composé : confidence + trend + momentum
        let score = data.confidence;
        if (data.trend?.strength) score += data.trend.strength * 0.3;
        if (data.momentum?.rsi_value) {
            const rsi = data.momentum.rsi_value;
            if (rsi < 30 || rsi > 70) score += 10;
        }
        return Math.min(score, 100);
    }

    renderSignals() {
        const container = document.getElementById('signalsContainer');
        
        if (this.signals.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; color: #9ca3af; padding: 40px 20px;">
                    <i class="bi bi-hourglass-split" style="font-size: 32px; opacity: 0.5;"></i>
                    <p style="margin-top: 12px;">Aucun signal fort détecté</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.signals.map(signal => `
            <div class="signal-card ${signal.signal.toLowerCase()}">
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

    updateServerTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('fr-FR');
        document.getElementById('serverTime').textContent = timeStr;
    }

    async quickTrade(action) {
        const confirmation = confirm(`⚠️ Confirmer ${action} ${this.currentTicker} ?`);
        if (!confirmation) return;
        
        // TODO: Intégration Alpaca API
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

// Init au chargement
document.addEventListener('DOMContentLoaded', () => {
    window.scalperPro = new ScalperProUltra();
});
