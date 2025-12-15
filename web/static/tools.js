/**
 * PLOUTOS TOOLS - JavaScript
 * 5 Tools: Screener, Backtest, Alerts, Heatmap, Portfolio
 */

// 🔍 SCREENER
async function runScreener() {
    const tickersInput = document.getElementById('screener-tickers').value;
    const tickers = tickersInput.split(',').map(t => t.trim());
    const resultsDiv = document.getElementById('screener-results');
    
    resultsDiv.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin text-2xl"></i><p class="mt-2">⌛ Scan en cours...</p></div>';
    
    try {
        const response = await fetch('/api/screener', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({tickers, period: '3mo'})
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ ${data.error}</div>`;
            return;
        }
        
        let html = '<h4 class="font-bold text-green-400 mb-2">🟢 Top BUY Opportunities</h4>';
        data.buy_opportunities.slice(0, 5).forEach(stock => {
            html += `
                <div class="stock-card rounded-lg p-3 mb-2">
                    <div class="font-bold">${stock.ticker}</div>
                    <div class="text-sm">Score: ${stock.score}/100 | RSI: ${stock.rsi.toFixed(1)}</div>
                    <div class="text-xs text-gray-400">${stock.recommendation} (${stock.confidence}%)</div>
                </div>
            `;
        });
        
        html += '<h4 class="font-bold text-red-400 mt-4 mb-2">🔴 Top SELL Signals</h4>';
        data.sell_signals.slice(0, 3).forEach(stock => {
            html += `
                <div class="stock-card sell rounded-lg p-3 mb-2">
                    <div class="font-bold">${stock.ticker}</div>
                    <div class="text-sm">Score: ${stock.score}/100 | RSI: ${stock.rsi.toFixed(1)}</div>
                    <div class="text-xs text-gray-400">${stock.recommendation}</div>
                </div>
            `;
        });
        
        resultsDiv.innerHTML = html;
        
    } catch (error) {
        resultsDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ Erreur: ${error}</div>`;
    }
}

// 📊 BACKTEST
async function runBacktest() {
    const ticker = document.getElementById('backtest-ticker').value;
    const strategy = document.getElementById('backtest-strategy').value;
    const resultsDiv = document.getElementById('backtest-results');
    
    resultsDiv.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin text-2xl"></i><p class="mt-2">⌛ Backtest en cours...</p></div>';
    
    try {
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker, strategy, period: '1y', params: {}})
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultsDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ ${data.error}</div>`;
            return;
        }
        
        const m = data.metrics;
        const colorReturn = m.total_return_pct > 0 ? 'positive' : 'negative';
        
        resultsDiv.innerHTML = `
            <div class="bg-green-900/30 p-4 rounded-lg">
                <h4 class="font-bold mb-3">${ticker} - ${strategy}</h4>
                <table class="w-full text-sm">
                    <tr class="border-b border-gray-700"><td class="py-2">Total Return</td><td class="text-right ${colorReturn} font-bold">${m.total_return_pct.toFixed(2)}%</td></tr>
                    <tr class="border-b border-gray-700"><td class="py-2">Win Rate</td><td class="text-right">${m.win_rate.toFixed(1)}%</td></tr>
                    <tr class="border-b border-gray-700"><td class="py-2">Sharpe Ratio</td><td class="text-right">${m.sharpe_ratio.toFixed(2)}</td></tr>
                    <tr class="border-b border-gray-700"><td class="py-2">Max Drawdown</td><td class="text-right negative">${m.max_drawdown_pct.toFixed(2)}%</td></tr>
                    <tr class="border-b border-gray-700"><td class="py-2">vs Buy & Hold</td><td class="text-right ${m.vs_buy_hold > 0 ? 'positive' : 'negative'}">${m.vs_buy_hold > 0 ? '+' : ''}${m.vs_buy_hold.toFixed(2)}%</td></tr>
                    <tr><td class="py-2">Trades</td><td class="text-right">${m.total_trades}</td></tr>
                </table>
            </div>
        `;
        
    } catch (error) {
        resultsDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ Erreur: ${error}</div>`;
    }
}

// 🔔 ALERTS
async function loadAlerts() {
    const listDiv = document.getElementById('alerts-list');
    listDiv.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin text-2xl"></i><p class="mt-2">⌛ Chargement...</p></div>';
    
    try {
        const response = await fetch('/api/alerts/list');
        const data = await response.json();
        
        if (data.error) {
            listDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ ${data.error}</div>`;
            return;
        }
        
        let html = '<h4 class="font-bold mb-2">📝 Règles Actives</h4>';
        
        if (data.rules.length === 0) {
            html += '<p class="text-gray-400 text-sm">Aucune alerte configurée</p>';
        } else {
            data.rules.forEach(rule => {
                html += `
                    <div class="stock-card rounded-lg p-3 mb-2">
                        <div class="font-bold">${rule.ticker}</div>
                        <div class="text-sm">${rule.condition} | Valeur: ${rule.value}</div>
                        <div class="text-xs text-gray-400">Actif: ${rule.active ? '✅' : '❌'} | Triggers: ${rule.trigger_count}</div>
                    </div>
                `;
            });
        }
        
        listDiv.innerHTML = html;
        
    } catch (error) {
        listDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ Erreur: ${error}</div>`;
    }
}

async function addAlert() {
    const ticker = document.getElementById('alert-ticker').value;
    const condition = document.getElementById('alert-condition').value;
    const value = parseFloat(document.getElementById('alert-value').value);
    
    if (!ticker || !value) {
        alert('Remplissez tous les champs');
        return;
    }
    
    try {
        const response = await fetch('/api/alerts/add', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker, condition, value})
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Erreur: ' + data.error);
        } else {
            alert('✅ Alerte ajoutée !');
            loadAlerts();
            document.getElementById('alert-ticker').value = '';
            document.getElementById('alert-value').value = '';
        }
        
    } catch (error) {
        alert('Erreur: ' + error);
    }
}

// 🔥 HEATMAP
async function generateHeatmap() {
    const container = document.getElementById('heatmap-container');
    const insightsDiv = document.getElementById('heatmap-insights');
    
    container.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin text-2xl"></i><p class="mt-2">⌛ Génération heatmap...</p></div>';
    
    try {
        const response = await fetch('/api/correlation/heatmap', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({tickers: ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'SPY', 'QQQ'], period: '6mo'})
        });
        
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ ${data.error}</div>`;
            return;
        }
        
        // Créer la heatmap avec Plotly
        const tickers = data.tickers;
        const zValues = [];
        
        for (let i = 0; i < tickers.length; i++) {
            const row = [];
            for (let j = 0; j < tickers.length; j++) {
                const point = data.matrix.find(m => m.x === tickers[j] && m.y === tickers[i]);
                row.push(point ? point.value : 0);
            }
            zValues.push(row);
        }
        
        const heatmapData = [{
            z: zValues,
            x: tickers,
            y: tickers,
            type: 'heatmap',
            colorscale: 'RdYlGn',
            zmid: 0
        }];
        
        const layout = {
            title: 'Matrice de Corrélations',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {color: 'white'}
        };
        
        Plotly.newPlot('heatmap-container', heatmapData, layout);
        
        // Insights
        let insightsHtml = '<h4 class="font-bold mb-2">💡 Insights</h4>';
        data.insights.forEach(insight => {
            insightsHtml += `<div class="stock-card rounded-lg p-3 mb-2 text-sm">${insight}</div>`;
        });
        
        insightsDiv.innerHTML = insightsHtml;
        
    } catch (error) {
        container.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ Erreur: ${error}</div>`;
    }
}

// 💼 PORTFOLIO
async function loadPortfolio() {
    const summaryDiv = document.getElementById('portfolio-summary');
    summaryDiv.innerHTML = '<div class="text-center py-4"><i class="fas fa-spinner fa-spin text-2xl"></i><p class="mt-2">⌛ Chargement...</p></div>';
    
    try {
        const response = await fetch('/api/portfolio/summary');
        const data = await response.json();
        
        if (data.error) {
            summaryDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ ${data.error}</div>`;
            return;
        }
        
        const plColor = data.total_pl >= 0 ? 'positive' : 'negative';
        
        let html = `
            <div class="bg-green-900/30 p-4 rounded-lg mb-4">
                <h4 class="font-bold mb-3">📊 Résumé Global</h4>
                <table class="w-full text-sm">
                    <tr class="border-b border-gray-700"><td class="py-2">Valeur Totale</td><td class="text-right font-bold">$${data.total_value.toLocaleString()}</td></tr>
                    <tr class="border-b border-gray-700"><td class="py-2">Coût Total</td><td class="text-right">$${data.total_cost.toLocaleString()}</td></tr>
                    <tr><td class="py-2">P/L</td><td class="text-right ${plColor} font-bold">${data.total_pl >= 0 ? '+' : ''}$${data.total_pl.toLocaleString()} (${data.total_pl_pct.toFixed(2)}%)</td></tr>
                </table>
            </div>
            <h4 class="font-bold mb-2">📈 Positions</h4>
        `;
        
        if (data.positions && data.positions.length > 0) {
            data.positions.forEach(pos => {
                const posColor = pos.pl >= 0 ? 'positive' : 'negative';
                html += `
                    <div class="stock-card rounded-lg p-3 mb-2 relative">
                        <div class="absolute top-2 right-2">
                            <button onclick="deletePosition('${pos.ticker}')" class="btn-danger px-2 py-1 rounded text-xs">🗑️</button>
                        </div>
                        <div class="font-bold">${pos.ticker}</div>
                        <div class="text-sm">${pos.shares} shares @ $${pos.current_price}</div>
                        <div class="text-sm">P/L: <span class="${posColor}">${pos.pl >= 0 ? '+' : ''}$${pos.pl.toFixed(2)} (${pos.pl_pct.toFixed(2)}%)</span></div>
                        <div class="text-xs text-gray-400">Allocation: ${pos.allocation_pct.toFixed(1)}%</div>
                    </div>
                `;
            });
        } else {
            html += '<p class="text-gray-400 text-sm">Aucune position</p>';
        }
        
        summaryDiv.innerHTML = html;
        
    } catch (error) {
        summaryDiv.innerHTML = `<div class="bg-red-900/30 p-4 rounded-lg">❌ Erreur: ${error}</div>`;
    }
}

async function addPosition() {
    const ticker = document.getElementById('portfolio-ticker').value;
    const shares = parseFloat(document.getElementById('portfolio-shares').value);
    const avg_price = parseFloat(document.getElementById('portfolio-price').value);
    
    if (!ticker || !shares || !avg_price) {
        alert('Remplissez tous les champs');
        return;
    }
    
    try {
        const response = await fetch('/api/portfolio/add', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker, shares, avg_price})
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Erreur: ' + data.error);
        } else {
            alert('✅ Position ajoutée !');
            loadPortfolio();
            document.getElementById('portfolio-ticker').value = '';
            document.getElementById('portfolio-shares').value = '';
            document.getElementById('portfolio-price').value = '';
        }
    } catch (error) {
        alert('Erreur: ' + error);
    }
}

async function deletePosition(ticker) {
    if (!confirm(`⚠️ Supprimer ${ticker} du portfolio ?`)) {
        return;
    }
    
    try {
        const response = await fetch('/api/portfolio/remove', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ticker})
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('❌ Erreur: ' + data.error);
        } else {
            alert(`✅ ${ticker} supprimé !`);
            loadPortfolio();
        }
    } catch (error) {
        alert('❌ Erreur: ' + error);
    }
}

// Auto-load au chargement de la page
window.addEventListener('load', () => {
    loadAlerts();
    loadPortfolio();
    generateHeatmap();
});
