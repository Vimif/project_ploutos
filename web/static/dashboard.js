/**
 * PLOUTOS DASHBOARD V8 - JavaScript
 * 
 * Gestion complète du dashboard avec:
 * - Onglets Dashboard, V8 Oracle, V7 Legacy
 * - Prédictions multi-horizon
 * - Graphiques temps réel
 * - Recommandations de trading
 */

// Configuration
const API_BASE = window.location.origin;
const REFRESH_INTERVAL = 10000; // 10 secondes

// ========== TAB MANAGEMENT ==========
document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.add('hidden');
        });
        
        // Deactivate all buttons
        document.querySelectorAll('.tab-button').forEach(b => {
            b.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(tabName + '-tab').classList.remove('hidden');
        btn.classList.add('active');
    });
});

// Portfolio Chart
let portfolioChart = null;

function initCharts() {
    const ctx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: false,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#9CA3AF' }
                },
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#9CA3AF' }
                }
            }
        }
    });
}

// Format money
function formatMoney(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

// Format percentage
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

// ========== DASHBOARD FUNCTIONS ==========

async function updateAccount() {
    try {
        const res = await fetch(`${API_BASE}/api/account`);
        const data = await res.json();
        
        if (!data.error) {
            document.getElementById('portfolio-value').textContent = formatMoney(data.portfolio_value);
            document.getElementById('cash-value').textContent = formatMoney(data.cash);
            document.getElementById('buying-power').textContent = 'Buying Power: ' + formatMoney(data.buying_power);
        }
    } catch (error) {
        console.error('Error fetching account:', error);
    }
}

async function updatePositions() {
    try {
        const res = await fetch(`${API_BASE}/api/positions`);
        const positions = await res.json();
        
        const container = document.getElementById('positions-list');
        
        if (positions.length === 0) {
            container.innerHTML = '<p class="text-gray-400 text-sm">Aucune position</p>';
            return;
        }
        
        container.innerHTML = positions.map(pos => `
            <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                <div>
                    <span class="font-bold">${pos.symbol}</span>
                    <span class="text-gray-400 text-sm ml-2">${pos.qty} shares</span>
                </div>
                <div class="text-right">
                    <div class="font-bold ${pos.unrealized_pl >= 0 ? 'text-green-500' : 'text-red-500'}">
                        ${formatMoney(pos.unrealized_pl)}
                    </div>
                    <div class="text-xs text-gray-400">
                        ${formatPercent(pos.unrealized_plpc)}
                    </div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('Error fetching positions:', error);
    }
}

async function updateTrades() {
    try {
        const res = await fetch(`${API_BASE}/api/trades?days=3`);
        const trades = await res.json();
        
        const container = document.getElementById('trades-list');
        
        if (trades.length === 0) {
            container.innerHTML = '<p class="text-gray-400 text-sm">Aucun trade</p>';
            return;
        }
        
        container.innerHTML = trades.slice(0, 10).map(trade => {
            const color = trade.action === 'BUY' ? 'text-green-500' : 'text-red-500';
            const icon = trade.action === 'BUY' ? 'fa-arrow-up' : 'fa-arrow-down';
            const time = new Date(trade.timestamp).toLocaleString('fr-FR');
            
            return `
                <div class="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <div class="flex items-center space-x-2">
                        <i class="fas ${icon} ${color}"></i>
                        <span class="font-bold">${trade.symbol}</span>
                        <span class="text-gray-400 text-xs">${time}</span>
                    </div>
                    <span class="font-bold">${formatMoney(trade.amount)}</span>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Error fetching trades:', error);
    }
}

// ========== V8 ORACLE FUNCTIONS ==========

async function predictV8Ticker() {
    const ticker = document.getElementById('v8-ticker').value.trim().toUpperCase();
    if (!ticker) return;
    
    try {
        const res = await fetch(`${API_BASE}/api/v8/predict/${ticker}`);
        const data = await res.json();
        
        if (data.error) {
            alert('❌ ' + data.error);
            return;
        }
        
        // Afficher prédictions par horizon
        const predictions = data.predictions;
        
        // Court terme (intraday)
        if (predictions.intraday && !predictions.intraday.error) {
            const p = predictions.intraday;
            const signalClass = p.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
            document.getElementById('v8-intraday').innerHTML = `
                <div class="text-2xl font-bold ${signalClass} mb-2">${p.prediction}</div>
                <div class="text-sm text-gray-300 mb-3">Confiance: ${p.confidence.toFixed(1)}%</div>
                <div class="w-full bg-gray-700 rounded-full h-3">
                    <div class="confidence-bar ${p.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                         style="width: ${p.confidence}%"></div>
                </div>
                <div class="mt-3 text-xs text-gray-400">Prix: $${p.current_price.toFixed(2)}</div>
            `;
        } else {
            document.getElementById('v8-intraday').innerHTML = '<p class="text-red-500">Non disponible</p>';
        }
        
        // Moyen terme (weekly)
        if (predictions.weekly && !predictions.weekly.error) {
            const p = predictions.weekly;
            const signalClass = p.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
            document.getElementById('v8-weekly').innerHTML = `
                <div class="text-2xl font-bold ${signalClass} mb-2">${p.prediction}</div>
                <div class="text-sm text-gray-300 mb-3">Confiance: ${p.confidence.toFixed(1)}%</div>
                <div class="w-full bg-gray-700 rounded-full h-3">
                    <div class="confidence-bar ${p.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                         style="width: ${p.confidence}%"></div>
                </div>
                <div class="mt-3 text-xs text-gray-400">Horizon: ${p.horizon}</div>
            `;
        } else {
            document.getElementById('v8-weekly').innerHTML = '<p class="text-red-500">Non disponible</p>';
        }
        
        // Ensemble
        if (data.ensemble && !data.ensemble.error) {
            const e = data.ensemble;
            const signalClass = e.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
            document.getElementById('v8-ensemble').innerHTML = `
                <div class="text-3xl font-bold ${signalClass} mb-2">${e.prediction}</div>
                <div class="text-sm text-gray-300 mb-3">Confiance: ${e.confidence.toFixed(1)}%</div>
                <div class="w-full bg-gray-700 rounded-full h-3 mb-3">
                    <div class="confidence-bar ${e.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                         style="width: ${e.confidence}%"></div>
                </div>
                <div class="text-xs font-semibold">
                    Agreement: <span class="${e.agreement === 'STRONG' ? 'text-green-400' : 'text-yellow-400'}">
                        ${e.agreement}
                    </span>
                </div>
                <div class="text-xs text-gray-400 mt-1">${e.models_used} modèles</div>
            `;
        } else {
            document.getElementById('v8-ensemble').innerHTML = '<p class="text-gray-400">1 modèle seul</p>';
        }
        
        // Recommandation
        const risk = document.getElementById('v8-risk').value;
        await getV8Recommendation(ticker, risk);
        
    } catch (error) {
        console.error('Error V8 prediction:', error);
        alert('❌ Erreur: ' + error.message);
    }
}

async function getV8Recommendation(ticker, risk) {
    try {
        const res = await fetch(`${API_BASE}/api/v8/recommend/${ticker}?risk=${risk}`);
        const data = await res.json();
        
        if (data.error) return;
        
        const actionClass = data.action === 'BUY' ? 'signal-buy' : data.action === 'SELL' ? 'signal-sell' : 'signal-hold';
        const strengthIcon = data.strength === 'STRONG' ? 'fa-star' : 'fa-star-half-alt';
        
        document.getElementById('v8-recommendation').innerHTML = `
            <div class="space-y-4">
                <div>
                    <h5 class="text-2xl font-bold mb-2">${ticker}</h5>
                    <div class="signal-badge ${actionClass}">
                        <i class="fas ${strengthIcon} mr-2"></i>
                        ${data.strength} ${data.action}
                    </div>
                </div>
                <div class="grid grid-cols-2 gap-4 text-left">
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-gray-400 text-sm">Confiance</div>
                        <div class="text-xl font-bold">${data.confidence.toFixed(1)}%</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-gray-400 text-sm">Seuil Utilisé</div>
                        <div class="text-xl font-bold">${data.threshold_used}%</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-gray-400 text-sm">Prédiction</div>
                        <div class="text-xl font-bold">${data.prediction}</div>
                    </div>
                    <div class="bg-gray-700 rounded p-3">
                        <div class="text-gray-400 text-sm">Agreement</div>
                        <div class="text-xl font-bold">${data.agreement || 'N/A'}</div>
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Error V8 recommendation:', error);
    }
}

async function analyzeV8Batch() {
    try {
        const res = await fetch(`${API_BASE}/api/v8/batch?tickers=NVDA,MSFT,AAPL,GOOGL,AMZN,META,TSLA`);
        const data = await res.json();
        
        if (data.error) {
            alert('❌ ' + data.error);
            return;
        }
        
        const container = document.getElementById('v8-batch-results');
        const tickers = data.tickers;
        
        container.innerHTML = '';
        
        for (const [ticker, result] of Object.entries(tickers)) {
            if (result.error) continue;
            
            const ensemble = result.ensemble;
            if (!ensemble || ensemble.error) continue;
            
            const signalClass = ensemble.prediction === 'UP' ? 'bg-green-900/30 border-green-500' : 'bg-red-900/30 border-red-500';
            const textClass = ensemble.prediction === 'UP' ? 'text-green-400' : 'text-red-400';
            
            container.innerHTML += `
                <div class="${signalClass} border rounded-lg p-4">
                    <div class="flex justify-between items-start mb-2">
                        <h5 class="font-bold text-lg">${ticker}</h5>
                        <span class="text-xs px-2 py-1 rounded ${ensemble.agreement === 'STRONG' ? 'bg-green-600' : 'bg-yellow-600'}">
                            ${ensemble.agreement}
                        </span>
                    </div>
                    <div class="text-2xl font-bold ${textClass} mb-2">${ensemble.prediction}</div>
                    <div class="text-sm text-gray-300 mb-2">Confiance: ${ensemble.confidence.toFixed(1)}%</div>
                    <div class="w-full bg-gray-700 rounded-full h-2">
                        <div class="confidence-bar ${ensemble.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                             style="width: ${ensemble.confidence}%"></div>
                    </div>
                </div>
            `;
        }
        
        // Summary
        if (data.summary) {
            const s = data.summary;
            container.innerHTML += `
                <div class="col-span-full bg-blue-900/30 border border-blue-500 rounded-lg p-4">
                    <h5 class="font-bold mb-2">Résumé</h5>
                    <div class="grid grid-cols-3 gap-2 text-sm">
                        <div>Bullish: <span class="font-bold text-green-400">${s.bullish}</span></div>
                        <div>Bearish: <span class="font-bold text-red-400">${s.bearish}</span></div>
                        <div>High Conf: <span class="font-bold text-yellow-400">${s.high_confidence_count}</span></div>
                    </div>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error V8 batch:', error);
        alert('❌ Erreur: ' + error.message);
    }
}

// ========== V7 LEGACY FUNCTIONS ==========

async function analyzeV7Ticker() {
    const ticker = document.getElementById('v7-ticker').value.trim().toUpperCase();
    if (!ticker) return;
    
    try {
        const res = await fetch(`${API_BASE}/api/v7/analysis?ticker=${ticker}`);
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('v7-detail').innerHTML = `<p class="text-red-500">❌ ${data.error}</p>`;
            return;
        }
        
        const signalClass = data.signal === 'BUY' ? 'signal-buy' : data.signal === 'SELL' ? 'signal-sell' : 'signal-hold';
        
        document.getElementById('v7-detail').innerHTML = `
            <div class="space-y-4">
                <h4 class="text-2xl font-bold">${ticker}</h4>
                <div class="signal-badge ${signalClass}">${data.strength} ${data.signal}</div>
                
                <div class="space-y-2 text-left">
                    <div class="flex justify-between p-2 bg-gray-700 rounded">
                        <span>Momentum</span>
                        <span class="font-bold">${data.experts.momentum.prediction} (${data.experts.momentum.confidence.toFixed(1)}%)</span>
                    </div>
                    <div class="flex justify-between p-2 bg-gray-700 rounded">
                        <span>Reversion</span>
                        <span class="font-bold">${data.experts.reversion.prediction} (${data.experts.reversion.confidence.toFixed(1)}%)</span>
                    </div>
                    <div class="flex justify-between p-2 bg-gray-700 rounded">
                        <span>Volatility</span>
                        <span class="font-bold">${data.experts.volatility.prediction} (${data.experts.volatility.confidence.toFixed(1)}%)</span>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('v7-momentum').textContent = `${data.experts.momentum.prediction} - ${data.experts.momentum.confidence.toFixed(1)}%`;
        document.getElementById('v7-reversion').textContent = `${data.experts.reversion.prediction} - ${data.experts.reversion.confidence.toFixed(1)}%`;
        document.getElementById('v7-volatility').textContent = `${data.experts.volatility.prediction} - ${data.experts.volatility.confidence.toFixed(1)}%`;
        
    } catch (error) {
        document.getElementById('v7-detail').innerHTML = `<p class="text-red-500">❌ ${error.message}</p>`;
    }
}

async function analyzeV7Batch() {
    try {
        const res = await fetch(`${API_BASE}/api/v7/batch?tickers=NVDA,AAPL,MSFT,GOOGL,AMZN`);
        const data = await res.json();
        
        if (!data.results || data.results.length === 0) {
            document.getElementById('v7-batch-results').innerHTML = '<p class="text-red-500">Aucun résultat</p>';
            return;
        }
        
        document.getElementById('v7-batch-results').innerHTML = data.results.map(r => {
            const signalClass = r.signal === 'BUY' ? 'text-green-500' : r.signal === 'SELL' ? 'text-red-500' : 'text-yellow-500';
            return `
                <div class="p-3 bg-gray-700 rounded flex justify-between items-center">
                    <div>
                        <span class="font-bold">${r.ticker}</span>
                        <span class="text-gray-400 text-sm ml-2">${r.strength}</span>
                    </div>
                    <span class="font-bold ${signalClass}">${r.signal}</span>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        document.getElementById('v7-batch-results').innerHTML = `<p class="text-red-500">❌ ${error.message}</p>`;
    }
}

// ========== EVENT LISTENERS ==========

// V8 Oracle
document.getElementById('v8-predict-btn').addEventListener('click', predictV8Ticker);
document.getElementById('v8-batch-btn').addEventListener('click', analyzeV8Batch);
document.getElementById('v8-ticker').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') predictV8Ticker();
});

// V7 Legacy
document.getElementById('v7-search-btn').addEventListener('click', analyzeV7Ticker);
document.getElementById('v7-batch-btn').addEventListener('click', analyzeV7Batch);
document.getElementById('v7-ticker').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeV7Ticker();
});

// Update all data
async function updateAll() {
    await Promise.all([
        updateAccount(),
        updatePositions(),
        updateTrades()
    ]);
    
    document.getElementById('last-update').textContent = 
        'Mis à jour: ' + new Date().toLocaleTimeString('fr-FR');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    updateAll();
    setInterval(updateAll, REFRESH_INTERVAL);
});
