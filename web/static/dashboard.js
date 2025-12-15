/**
 * PLOUTOS DASHBOARD V8 - JavaScript
 * 
 * Gestion complète du dashboard avec:
 * - Watchlists intégrées (clic pour analyser)
 * - Prédictions multi-horizon V8 Oracle
 * - Affichage temps réel des tickers
 * - Recommandations de trading
 */

// Configuration
const API_BASE = window.location.origin;
const REFRESH_INTERVAL = 10000; // 10 secondes

// État global
let selectedTickers = [];
let currentWatchlist = null;

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
    const ctx = document.getElementById('portfolio-chart');
    if (!ctx) return;
    
    portfolioChart = new Chart(ctx.getContext('2d'), {
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

// ========== WATCHLIST INTEGRATION ==========

// Écouter les sélections de watchlists
window.addEventListener('watchlistSelected', async (event) => {
    const { listId, name, tickers } = event.detail;
    
    console.log(`📊 Watchlist sélectionnée: ${name}`, tickers);
    
    selectedTickers = tickers;
    currentWatchlist = name;
    
    // Afficher dans la zone principale
    displayWatchlistTickers(name, tickers);
    
    // Auto-analyse batch si pas trop de tickers
    if (tickers.length <= 10) {
        await analyzeWatchlistBatch(tickers);
    }
});

function displayWatchlistTickers(name, tickers) {
    // Trouver la zone d'affichage principale
    let container = document.getElementById('watchlist-tickers-display');
    
    // Si elle n'existe pas, la créer
    if (!container) {
        const mainContent = document.getElementById('main-content');
        if (!mainContent) return;
        
        // Créer la section d'affichage des tickers
        const section = document.createElement('div');
        section.id = 'watchlist-tickers-display';
        section.className = 'mb-6';
        mainContent.insertBefore(section, mainContent.firstChild);
        container = section;
    }
    
    container.innerHTML = `
        <div class="bg-gray-800 rounded-lg p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-bold">
                    📊 ${name}
                    <span class="text-sm text-gray-400 ml-2">(${tickers.length} tickers)</span>
                </h3>
                <button onclick="analyzeWatchlistBatch(selectedTickers)" 
                        class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm">
                    <i class="fas fa-chart-line mr-2"></i>Analyser tout
                </button>
            </div>
            
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3" id="tickers-grid">
                ${tickers.map(ticker => `
                    <button onclick="quickAnalyzeTicker('${ticker}')" 
                            class="ticker-card bg-gray-700 hover:bg-gray-600 p-4 rounded-lg text-center transition-all">
                        <div class="font-bold text-lg">${ticker}</div>
                        <div class="text-xs text-gray-400 mt-1">Cliquer pour analyser</div>
                    </button>
                `).join('')}
            </div>
            
            <div id="quick-analysis-result" class="mt-4"></div>
        </div>
    `;
}

async function quickAnalyzeTicker(ticker) {
    console.log(`🔍 Analyse rapide de ${ticker}`);
    
    const resultDiv = document.getElementById('quick-analysis-result');
    if (!resultDiv) return;
    
    resultDiv.innerHTML = `
        <div class="bg-gray-700 rounded p-4 animate-pulse">
            <p class="text-center">⏳ Analyse de ${ticker} en cours...</p>
        </div>
    `;
    
    try {
        // Appeler l'API V8 Oracle si disponible
        const res = await fetch(`${API_BASE}/api/v8/predict/${ticker}`);
        const data = await res.json();
        
        if (data.error) {
            resultDiv.innerHTML = `
                <div class="bg-red-900/30 border border-red-500 rounded p-4">
                    <p class="text-red-400">❌ ${data.error}</p>
                    <p class="text-xs text-gray-400 mt-2">Essayez le module Chart pour plus de détails</p>
                </div>
            `;
            return;
        }
        
        const ensemble = data.ensemble || {};
        const prediction = ensemble.prediction || 'N/A';
        const confidence = ensemble.confidence || 0;
        const agreement = ensemble.agreement || 'WEAK';
        
        const signalClass = prediction === 'UP' ? 'bg-green-900/30 border-green-500' : 'bg-red-900/30 border-red-500';
        const textClass = prediction === 'UP' ? 'text-green-400' : 'text-red-400';
        const icon = prediction === 'UP' ? 'fa-arrow-up' : 'fa-arrow-down';
        
        resultDiv.innerHTML = `
            <div class="${signalClass} border rounded-lg p-6">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h4 class="text-2xl font-bold">${ticker}</h4>
                        <p class="text-sm text-gray-400">Prédiction V8 Oracle</p>
                    </div>
                    <div class="text-right">
                        <div class="text-3xl font-bold ${textClass}">
                            <i class="fas ${icon} mr-2"></i>${prediction}
                        </div>
                        <div class="text-sm text-gray-300">Confiance: ${confidence.toFixed(1)}%</div>
                    </div>
                </div>
                
                <div class="w-full bg-gray-700 rounded-full h-3 mb-4">
                    <div class="${prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'} h-3 rounded-full transition-all" 
                         style="width: ${confidence}%"></div>
                </div>
                
                <div class="grid grid-cols-3 gap-4 text-center">
                    <div class="bg-gray-800 rounded p-3">
                        <div class="text-xs text-gray-400">Agreement</div>
                        <div class="font-bold ${agreement === 'STRONG' ? 'text-green-400' : 'text-yellow-400'}">
                            ${agreement}
                        </div>
                    </div>
                    <div class="bg-gray-800 rounded p-3">
                        <div class="text-xs text-gray-400">Modèles</div>
                        <div class="font-bold">${ensemble.models_used || 1}</div>
                    </div>
                    <div class="bg-gray-800 rounded p-3">
                        <a href="/chart?ticker=${ticker}" class="text-blue-400 hover:text-blue-300 text-sm">
                            <i class="fas fa-chart-area mr-1"></i>Voir graphique
                        </a>
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Erreur analyse ticker:', error);
        resultDiv.innerHTML = `
            <div class="bg-red-900/30 border border-red-500 rounded p-4">
                <p class="text-red-400">❌ Erreur: ${error.message}</p>
            </div>
        `;
    }
}

async function analyzeWatchlistBatch(tickers) {
    console.log('📊 Analyse batch:', tickers);
    
    const resultDiv = document.getElementById('quick-analysis-result');
    if (!resultDiv) return;
    
    resultDiv.innerHTML = `
        <div class="bg-gray-700 rounded p-4 animate-pulse">
            <p class="text-center">⏳ Analyse de ${tickers.length} tickers en cours...</p>
        </div>
    `;
    
    try {
        const tickersParam = tickers.join(',');
        const res = await fetch(`${API_BASE}/api/v8/batch?tickers=${tickersParam}`);
        const data = await res.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p class="text-red-500">❌ ${data.error}</p>`;
            return;
        }
        
        const results = data.tickers || {};
        const summary = data.summary || { bullish: 0, bearish: 0 };
        
        let html = `
            <div class="space-y-3">
                <div class="bg-blue-900/30 border border-blue-500 rounded p-3">
                    <div class="flex justify-between text-sm">
                        <span>🟢 Bullish: <strong class="text-green-400">${summary.bullish}</strong></span>
                        <span>🔴 Bearish: <strong class="text-red-400">${summary.bearish}</strong></span>
                        <span>⚡ High Conf: <strong class="text-yellow-400">${summary.high_confidence_count || 0}</strong></span>
                    </div>
                </div>
                
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        `;
        
        for (const [ticker, result] of Object.entries(results)) {
            if (result.error) continue;
            
            const ens = result.ensemble || {};
            const pred = ens.prediction || 'N/A';
            const conf = ens.confidence || 0;
            
            const bgClass = pred === 'UP' ? 'bg-green-900/20 border-green-500' : 'bg-red-900/20 border-red-500';
            const textClass = pred === 'UP' ? 'text-green-400' : 'text-red-400';
            
            html += `
                <div class="${bgClass} border rounded p-3 cursor-pointer hover:scale-105 transition-transform" 
                     onclick="quickAnalyzeTicker('${ticker}')">
                    <div class="font-bold text-sm mb-1">${ticker}</div>
                    <div class="text-xl font-bold ${textClass}">${pred}</div>
                    <div class="text-xs text-gray-400">${conf.toFixed(0)}%</div>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
        
        resultDiv.innerHTML = html;
        
    } catch (error) {
        console.error('Erreur batch:', error);
        resultDiv.innerHTML = `<p class="text-red-500">❌ ${error.message}</p>`;
    }
}

// ========== DASHBOARD FUNCTIONS ==========

async function updateAccount() {
    try {
        const res = await fetch(`${API_BASE}/api/account`);
        const data = await res.json();
        
        if (!data.error) {
            document.getElementById('portfolio-value').textContent = formatMoney(data.portfolio_value || 0);
            document.getElementById('cash-value').textContent = formatMoney(data.cash || 0);
            document.getElementById('buying-power').textContent = 'Buying Power: ' + formatMoney(data.buying_power || 0);
        }
    } catch (error) {
        console.error('Error fetching account:', error);
    }
}

async function updatePositions() {
    try {
        const res = await fetch(`${API_BASE}/api/positions`);
        const data = await res.json();
        
        const positions = data.positions || [];
        const container = document.getElementById('positions-list');
        if (!container) return;
        
        if (positions.length === 0) {
            container.innerHTML = '<p class="text-gray-400 text-sm">Aucune position</p>';
            return;
        }
        
        container.innerHTML = positions.map(pos => `
            <div class="flex justify-between items-center p-2 bg-gray-700 rounded hover:bg-gray-600 cursor-pointer"
                 onclick="quickAnalyzeTicker('${pos.symbol}')">
                <div>
                    <span class="font-bold">${pos.symbol}</span>
                    <span class="text-gray-400 text-sm ml-2">${pos.qty} shares</span>
                </div>
                <div class="text-right">
                    <div class="font-bold ${pos.unrealized_pl >= 0 ? 'text-green-500' : 'text-red-500'}">
                        ${formatMoney(pos.unrealized_pl)}
                    </div>
                    <div class="text-xs text-gray-400">
                        ${formatPercent(pos.unrealized_plpc / 100)}
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
        const data = await res.json();
        
        const trades = data.trades || [];
        const container = document.getElementById('trades-list');
        if (!container) return;
        
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
                        <span class="font-bold">${trade.ticker}</span>
                        <span class="text-gray-400 text-xs">${time}</span>
                    </div>
                    <span class="font-bold">${formatMoney(trade.price * trade.quantity)}</span>
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
    if (!ticker) {
        alert('⚠️  Veuillez entrer un ticker');
        return;
    }
    
    try {
        const res = await fetch(`${API_BASE}/api/v8/predict/${ticker}`);
        const data = await res.json();
        
        console.log('V8 Response:', data);
        
        if (data.error) {
            alert('❌ ' + data.error);
            document.getElementById('v8-intraday').innerHTML = '<p class="text-red-500">' + data.error + '</p>';
            document.getElementById('v8-weekly').innerHTML = '<p class="text-red-500">' + data.error + '</p>';
            document.getElementById('v8-ensemble').innerHTML = '<p class="text-red-500">' + data.error + '</p>';
            return;
        }
        
        const predictions = data.predictions || {};
        
        // Court terme (intraday)
        const intradayEl = document.getElementById('v8-intraday');
        if (intradayEl) {
            if (predictions.intraday && !predictions.intraday.error) {
                const p = predictions.intraday;
                const signalClass = p.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
                intradayEl.innerHTML = `
                    <div class="text-2xl font-bold ${signalClass} mb-2">${p.prediction}</div>
                    <div class="text-sm text-gray-300 mb-3">Confiance: ${p.confidence.toFixed(1)}%</div>
                    <div class="w-full bg-gray-700 rounded-full h-3">
                        <div class="confidence-bar ${p.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                             style="width: ${p.confidence}%"></div>
                    </div>
                    <div class="mt-3 text-xs text-gray-400">Prix: $${p.current_price.toFixed(2)}</div>
                `;
            } else {
                intradayEl.innerHTML = '<p class="text-yellow-500">Modèle non entraîné</p>';
            }
        }
        
        // Moyen terme (weekly)
        const weeklyEl = document.getElementById('v8-weekly');
        if (weeklyEl) {
            if (predictions.weekly && !predictions.weekly.error) {
                const p = predictions.weekly;
                const signalClass = p.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
                weeklyEl.innerHTML = `
                    <div class="text-2xl font-bold ${signalClass} mb-2">${p.prediction}</div>
                    <div class="text-sm text-gray-300 mb-3">Confiance: ${p.confidence.toFixed(1)}%</div>
                    <div class="w-full bg-gray-700 rounded-full h-3">
                        <div class="confidence-bar ${p.prediction === 'UP' ? 'bg-green-500' : 'bg-red-500'}" 
                             style="width: ${p.confidence}%"></div>
                    </div>
                    <div class="mt-3 text-xs text-gray-400">Horizon: ${p.horizon || '5 jours'}</div>
                `;
            } else {
                weeklyEl.innerHTML = '<p class="text-yellow-500">Modèle non entraîné</p>';
            }
        }
        
        // Ensemble
        const ensembleEl = document.getElementById('v8-ensemble');
        if (ensembleEl) {
            if (data.ensemble && !data.ensemble.error) {
                const e = data.ensemble;
                const signalClass = e.prediction === 'UP' ? 'text-green-500' : 'text-red-500';
                ensembleEl.innerHTML = `
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
                    <div class="text-xs text-gray-400 mt-1">${e.models_used || 1} modèle(s)</div>
                `;
            } else {
                ensembleEl.innerHTML = '<p class="text-gray-400">1 modèle unique</p>';
            }
        }
        
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
        
        const recEl = document.getElementById('v8-recommendation');
        if (!recEl) return;
        
        if (data.error) {
            recEl.innerHTML = `<p class="text-red-500">❌ ${data.error}</p>`;
            return;
        }
        
        const actionClass = data.action === 'BUY' ? 'signal-buy' : data.action === 'SELL' ? 'signal-sell' : 'signal-hold';
        const strengthIcon = data.strength === 'STRONG' ? 'fa-star' : 'fa-star-half-alt';
        
        recEl.innerHTML = `
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
        
        const container = document.getElementById('v8-batch-results');
        if (!container) return;
        
        if (data.error) {
            alert('❌ ' + data.error);
            container.innerHTML = `<p class="text-red-500">${data.error}</p>`;
            return;
        }
        
        const tickers = data.tickers || {};
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

// ========== EVENT LISTENERS ==========

document.addEventListener('DOMContentLoaded', () => {
    // V8 Oracle
    const v8PredictBtn = document.getElementById('v8-predict-btn');
    const v8BatchBtn = document.getElementById('v8-batch-btn');
    const v8Ticker = document.getElementById('v8-ticker');
    
    if (v8PredictBtn) v8PredictBtn.addEventListener('click', predictV8Ticker);
    if (v8BatchBtn) v8BatchBtn.addEventListener('click', analyzeV8Batch);
    if (v8Ticker) {
        v8Ticker.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') predictV8Ticker();
        });
    }
    
    // Initialize
    initCharts();
    updateAll();
    setInterval(updateAll, REFRESH_INTERVAL);
});

// Update all data
async function updateAll() {
    await Promise.all([
        updateAccount(),
        updatePositions(),
        updateTrades()
    ]);
    
    const lastUpdateEl = document.getElementById('last-update');
    if (lastUpdateEl) {
        lastUpdateEl.textContent = 'Mis à jour: ' + new Date().toLocaleTimeString('fr-FR');
    }
}
