/**
 * PLOUTOS ADVANCED CHART - JavaScript
 * Graphiques interactifs avec Plotly + Assistant IA
 */

const API_BASE = window.location.origin;
let currentData = null;
let activeIndicators = new Set(['sma']);

// ========== CHART FUNCTIONS ==========

async function analyzeStock() {
    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
    const period = document.getElementById('period-select').value;
    
    if (!ticker) {
        alert('⚠️ Veuillez entrer un ticker');
        return;
    }
    
    addAIMessage(`🔍 Analyse de ${ticker} sur ${period}...`, 'ai');
    
    try {
        const res = await fetch(`${API_BASE}/api/chart/${ticker}?period=${period}`);
        const data = await res.json();
        
        if (data.error) {
            addAIMessage(`❌ Erreur: ${data.error}`, 'ai');
            return;
        }
        
        currentData = data;
        renderMainChart(data);
        renderRSIChart(data);
        renderMACDChart(data);
        updateIndicatorsSummary(data);
        generateAIAnalysis(data);
        
    } catch (error) {
        console.error('Error:', error);
        addAIMessage(`❌ Erreur technique: ${error.message}`, 'ai');
    }
}

function renderMainChart(data) {
    const traces = [];
    
    // Candlestick
    traces.push({
        type: 'candlestick',
        x: data.dates,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        name: 'Prix',
        increasing: {line: {color: '#10b981'}},
        decreasing: {line: {color: '#ef4444'}}
    });
    
    // SMA
    if (activeIndicators.has('sma')) {
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.sma_20,
            name: 'SMA 20',
            line: {color: '#3b82f6', width: 2}
        });
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.sma_50,
            name: 'SMA 50',
            line: {color: '#f59e0b', width: 2}
        });
    }
    
    // Bollinger Bands
    if (activeIndicators.has('bb')) {
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.bb_upper,
            name: 'BB Upper',
            line: {color: '#8b5cf6', width: 1, dash: 'dot'},
            fill: 'tonexty',
            fillcolor: 'rgba(139, 92, 246, 0.1)'
        });
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.bb_lower,
            name: 'BB Lower',
            line: {color: '#8b5cf6', width: 1, dash: 'dot'}
        });
    }
    
    const layout = {
        title: `${data.ticker} - ${data.current_price.toFixed(2)} USD`,
        xaxis: {title: 'Date', gridcolor: '#374151'},
        yaxis: {title: 'Prix (USD)', gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af'},
        showlegend: true,
        legend: {x: 0, y: 1},
        margin: {l: 50, r: 50, t: 50, b: 50}
    };
    
    Plotly.newPlot('main-chart', traces, layout, {responsive: true});
}

function renderRSIChart(data) {
    const trace = {
        type: 'scatter',
        x: data.dates,
        y: data.indicators.rsi,
        name: 'RSI',
        line: {color: '#3b82f6', width: 2}
    };
    
    const layout = {
        xaxis: {title: '', gridcolor: '#374151'},
        yaxis: {title: 'RSI', range: [0, 100], gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af'},
        showlegend: false,
        margin: {l: 50, r: 50, t: 10, b: 30},
        shapes: [
            {type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], 
             y0: 70, y1: 70, line: {color: '#ef4444', width: 1, dash: 'dash'}},
            {type: 'line', x0: data.dates[0], x1: data.dates[data.dates.length-1], 
             y0: 30, y1: 30, line: {color: '#10b981', width: 1, dash: 'dash'}}
        ]
    };
    
    Plotly.newPlot('rsi-chart', [trace], layout, {responsive: true});
}

function renderMACDChart(data) {
    const traces = [
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.macd,
            name: 'MACD',
            line: {color: '#3b82f6', width: 2}
        },
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.macd_signal,
            name: 'Signal',
            line: {color: '#f59e0b', width: 2}
        },
        {
            type: 'bar',
            x: data.dates,
            y: data.indicators.macd_hist,
            name: 'Histogram',
            marker: {color: '#8b5cf6'}
        }
    ];
    
    const layout = {
        xaxis: {title: '', gridcolor: '#374151'},
        yaxis: {title: 'MACD', gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af'},
        showlegend: false,
        margin: {l: 50, r: 50, t: 10, b: 30}
    };
    
    Plotly.newPlot('macd-chart', traces, layout, {responsive: true});
}

// ========== INDICATORS SUMMARY ==========

function updateIndicatorsSummary(data) {
    const summary = data.analysis;
    const container = document.getElementById('indicators-summary');
    
    container.innerHTML = `
        <div class="space-y-2">
            <div class="indicator-badge bg-${summary.trend === 'BULLISH' ? 'green' : summary.trend === 'BEARISH' ? 'red' : 'yellow'}-900/30 
                        border border-${summary.trend === 'BULLISH' ? 'green' : summary.trend === 'BEARISH' ? 'red' : 'yellow'}-500 
                        rounded-lg p-3">
                <div class="flex justify-between items-center">
                    <span class="text-sm font-bold">Tendance</span>
                    <span class="text-lg font-bold">${summary.trend}</span>
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-3">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs text-gray-400">RSI (14)</span>
                    <span class="font-bold text-${summary.rsi > 70 ? 'red' : summary.rsi < 30 ? 'green' : 'yellow'}-400">
                        ${summary.rsi.toFixed(1)}
                    </span>
                </div>
                <div class="text-xs text-gray-400">
                    ${summary.rsi > 70 ? '🔴 Suracheté' : summary.rsi < 30 ? '🟢 Survendu' : '🟡 Neutre'}
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-3">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs text-gray-400">MACD</span>
                    <span class="font-bold text-${summary.macd_signal === 'BULLISH' ? 'green' : 'red'}-400">
                        ${summary.macd_signal}
                    </span>
                </div>
                <div class="text-xs text-gray-400">
                    ${summary.macd_crossover ? '⚡ Croisement détecté' : 'Pas de croisement'}
                </div>
            </div>
            
            <div class="bg-gray-700 rounded-lg p-3">
                <div class="flex justify-between items-center mb-1">
                    <span class="text-xs text-gray-400">Volatilité</span>
                    <span class="font-bold">${summary.volatility}</span>
                </div>
                <div class="text-xs text-gray-400">
                    BB Width: ${summary.bb_width.toFixed(2)}%
                </div>
            </div>
            
            <div class="bg-blue-900/30 border border-blue-500 rounded-lg p-3">
                <div class="text-xs text-gray-400 mb-1">Signal Global</div>
                <div class="text-lg font-bold text-${summary.overall_signal === 'BUY' ? 'green' : summary.overall_signal === 'SELL' ? 'red' : 'yellow'}-400">
                    ${summary.overall_signal}
                </div>
                <div class="text-xs text-gray-300 mt-1">
                    Confiance: ${summary.confidence.toFixed(0)}%
                </div>
            </div>
        </div>
    `;
}

// ========== AI ASSISTANT ==========

function generateAIAnalysis(data) {
    const analysis = data.analysis;
    
    // Message principal
    let message = `📊 **Analyse technique de ${data.ticker}**\n\n`;
    
    // Tendance
    if (analysis.trend === 'BULLISH') {
        message += `🟢 **Tendance haussière détectée**\n`;
        message += `Le prix est au-dessus des moyennes mobiles SMA 20 et SMA 50, indiquant une dynamique positive.\n\n`;
    } else if (analysis.trend === 'BEARISH') {
        message += `🔴 **Tendance baissière détectée**\n`;
        message += `Le prix est en-dessous des moyennes mobiles, indiquant une pression vendeuse.\n\n`;
    } else {
        message += `🟡 **Marché en consolidation**\n`;
        message += `Le prix oscille autour des moyennes mobiles sans direction claire.\n\n`;
    }
    
    // RSI
    message += `📊 **RSI (${analysis.rsi.toFixed(1)})**\n`;
    if (analysis.rsi > 70) {
        message += `⚠️ Zone de sur-achat ! Le titre pourrait connaître une correction prochainement.\n\n`;
    } else if (analysis.rsi < 30) {
        message += `🟢 Zone de sur-vente ! Opportunité d'achat potentielle si la tendance globale est positive.\n\n`;
    } else {
        message += `Zone neutre, le momentum n'est ni extrême haussier ni baissier.\n\n`;
    }
    
    // MACD
    if (analysis.macd_crossover) {
        message += `⚡ **Croisement MACD détecté !**\n`;
        message += `Signal ${analysis.macd_signal} - Ceci pourrait indiquer un changement de tendance.\n\n`;
    }
    
    // Recommandation finale
    message += `🎯 **Recommandation: ${analysis.overall_signal}**\n`;
    message += `Confiance: ${analysis.confidence.toFixed(0)}%\n\n`;
    
    if (analysis.overall_signal === 'BUY') {
        message += `Les indicateurs techniques sont majoritairement positifs. Envisagez une position longue avec stop-loss.`;
    } else if (analysis.overall_signal === 'SELL') {
        message += `Les indicateurs sont négatifs. Considérez de réduire l'exposition ou de shorter.`;
    } else {
        message += `Signaux mixtes. Attendez une confirmation avant d'entrer en position.`;
    }
    
    addAIMessage(message, 'ai');
}

function addAIMessage(text, sender = 'ai') {
    const container = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-message';
    
    const bgColor = sender === 'ai' ? 'bg-blue-900/30 border-blue-500' : 'bg-gray-700 border-gray-600';
    const icon = sender === 'ai' ? 'fa-robot text-blue-400' : 'fa-user text-gray-400';
    
    msgDiv.innerHTML = `
        <div class="${bgColor} border rounded-lg p-3">
            <div class="flex items-start gap-2">
                <i class="fas ${icon} mt-1"></i>
                <div class="flex-1 text-sm whitespace-pre-line">${text}</div>
            </div>
        </div>
    `;
    
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message || !currentData) return;
    
    addAIMessage(message, 'user');
    input.value = '';
    
    // Typing indicator
    addAIMessage('🤖 <span class="typing-indicator">•••</span>', 'ai');
    
    try {
        const res = await fetch(`${API_BASE}/api/ai-chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                ticker: currentData.ticker,
                context: currentData.analysis
            })
        });
        
        const data = await res.json();
        
        // Remove typing indicator
        const messages = document.getElementById('chat-messages');
        messages.removeChild(messages.lastChild);
        
        addAIMessage(data.response, 'ai');
        
    } catch (error) {
        console.error('Chat error:', error);
    }
}

// ========== EVENT LISTENERS ==========

document.getElementById('analyze-btn').addEventListener('click', analyzeStock);
document.getElementById('ticker-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeStock();
});

document.getElementById('chat-send').addEventListener('click', sendChatMessage);
document.getElementById('chat-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});

// Indicator toggles
document.querySelectorAll('.indicator-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        const indicator = btn.dataset.indicator;
        
        if (activeIndicators.has(indicator)) {
            activeIndicators.delete(indicator);
            btn.classList.remove('bg-blue-600');
            btn.classList.add('bg-gray-700');
        } else {
            activeIndicators.add(indicator);
            btn.classList.remove('bg-gray-700');
            btn.classList.add('bg-blue-600');
        }
        
        if (currentData) renderMainChart(currentData);
    });
});

// Auto-analyze on load
window.addEventListener('load', () => {
    analyzeStock();
});
