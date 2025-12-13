/**
 * PLOUTOS ADVANCED CHART - JavaScript V3
 * Compatible avec la nouvelle API (quick_stats + signals)
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
        
        console.log('📊 API Response:', data); // DEBUG
        
        if (data.error) {
            addAIMessage(`❌ Erreur: ${data.error}`, 'ai');
            return;
        }
        
        currentData = data;
        renderMainChart(data);
        renderRSIChart(data);
        renderMACDChart(data);
        renderStochChart(data);
        renderADXChart(data);
        renderATRChart(data);
        renderOBVChart(data);
        updateQuickStats(data);
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
    if (activeIndicators.has('sma') && data.indicators.sma_20) {
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
        if (data.indicators.sma_200) {
            traces.push({
                type: 'scatter',
                x: data.dates,
                y: data.indicators.sma_200,
                name: 'SMA 200',
                line: {color: '#ef4444', width: 1.5}
            });
        }
    }
    
    // EMA
    if (activeIndicators.has('ema') && data.indicators.ema_12) {
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.ema_12,
            name: 'EMA 12',
            line: {color: '#8b5cf6', width: 1.5, dash: 'dot'}
        });
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.ema_26,
            name: 'EMA 26',
            line: {color: '#ec4899', width: 1.5, dash: 'dot'}
        });
    }
    
    // Bollinger Bands
    if (activeIndicators.has('bb') && data.indicators.bb_upper) {
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
    
    // SAR
    if (activeIndicators.has('sar') && data.indicators.sar) {
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: data.dates,
            y: data.indicators.sar,
            name: 'SAR',
            marker: {color: '#fbbf24', size: 4}
        });
    }
    
    // VWAP
    if (activeIndicators.has('vwap') && data.indicators.vwap) {
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.vwap,
            name: 'VWAP',
            line: {color: '#a855f7', width: 2}
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
        legend: {x: 0, y: 1.1, orientation: 'h'},
        margin: {l: 50, r: 50, t: 50, b: 50}
    };
    
    Plotly.newPlot('main-chart', traces, layout, {responsive: true});
}

function renderRSIChart(data) {
    if (!data.indicators.rsi) return;
    
    const trace = {
        type: 'scatter',
        x: data.dates,
        y: data.indicators.rsi,
        name: 'RSI',
        line: {color: '#3b82f6', width: 2},
        fill: 'tozeroy'
    };
    
    const layout = {
        xaxis: {gridcolor: '#374151'},
        yaxis: {range: [0, 100], gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20},
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
    if (!data.indicators.macd) return;
    
    const traces = [
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.macd,
            name: 'MACD',
            line: {color: '#3b82f6', width: 1.5}
        },
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.macd_signal,
            name: 'Signal',
            line: {color: '#f59e0b', width: 1.5}
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
        xaxis: {gridcolor: '#374151'},
        yaxis: {gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20}
    };
    
    Plotly.newPlot('macd-chart', traces, layout, {responsive: true});
}

function renderStochChart(data) {
    if (!data.indicators.stoch_k) return;
    
    const traces = [
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.stoch_k,
            name: '%K',
            line: {color: '#3b82f6', width: 1.5}
        },
        {
            type: 'scatter',
            x: data.dates,
            y: data.indicators.stoch_d,
            name: '%D',
            line: {color: '#f59e0b', width: 1.5}
        }
    ];
    
    const layout = {
        xaxis: {gridcolor: '#374151'},
        yaxis: {range: [0, 100], gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20}
    };
    
    Plotly.newPlot('stoch-chart', traces, layout, {responsive: true});
}

function renderADXChart(data) {
    if (!data.indicators.adx) return;
    
    const trace = {
        type: 'scatter',
        x: data.dates,
        y: data.indicators.adx,
        line: {color: '#10b981', width: 2},
        fill: 'tozeroy'
    };
    
    const layout = {
        xaxis: {gridcolor: '#374151'},
        yaxis: {gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20}
    };
    
    Plotly.newPlot('adx-chart', [trace], layout, {responsive: true});
}

function renderATRChart(data) {
    if (!data.indicators.atr) return;
    
    const trace = {
        type: 'scatter',
        x: data.dates,
        y: data.indicators.atr,
        line: {color: '#ef4444', width: 2},
        fill: 'tozeroy'
    };
    
    const layout = {
        xaxis: {gridcolor: '#374151'},
        yaxis: {gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20}
    };
    
    Plotly.newPlot('atr-chart', [trace], layout, {responsive: true});
}

function renderOBVChart(data) {
    if (!data.indicators.obv) return;
    
    const trace = {
        type: 'scatter',
        x: data.dates,
        y: data.indicators.obv,
        line: {color: '#8b5cf6', width: 2}
    };
    
    const layout = {
        xaxis: {gridcolor: '#374151'},
        yaxis: {gridcolor: '#374151'},
        plot_bgcolor: '#1f2937',
        paper_bgcolor: '#1f2937',
        font: {color: '#9ca3af', size: 10},
        showlegend: false,
        margin: {l: 40, r: 10, t: 10, b: 20}
    };
    
    Plotly.newPlot('obv-chart', [trace], layout, {responsive: true});
}

// ========== QUICK STATS ==========

function updateQuickStats(data) {
    if (!data.quick_stats) return;
    
    const stats = data.quick_stats;
    const container = document.getElementById('quick-stats');
    
    container.innerHTML = `
        <div class="text-sm space-y-2">
            <div class="flex justify-between">
                <span class="text-gray-400">Prix</span>
                <span class="font-bold text-lg">${stats.price.toFixed(2)} $</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-400">Variation</span>
                <span class="${stats.change_pct >= 0 ? 'text-green-400' : 'text-red-400'} font-bold">
                    ${stats.change_pct >= 0 ? '+' : ''}${stats.change_pct.toFixed(2)}%
                </span>
            </div>
            <hr class="border-gray-700">
            <div class="flex justify-between">
                <span class="text-gray-400">RSI</span>
                <span class="${stats.rsi > 70 ? 'text-red-400' : stats.rsi < 30 ? 'text-green-400' : 'text-yellow-400'} font-bold">
                    ${stats.rsi.toFixed(1)}
                </span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-400">ADX</span>
                <span>${stats.adx.toFixed(1)}</span>
            </div>
            <div class="flex justify-between">
                <span class="text-gray-400">Volume</span>
                <span class="text-xs">${formatVolume(stats.volume)}</span>
            </div>
            <hr class="border-gray-700">
            <div class="p-2 rounded text-center font-bold ${
                stats.recommendation.includes('BUY') ? 'bg-green-600' : 
                stats.recommendation.includes('SELL') ? 'bg-red-600' : 'bg-gray-600'
            }">
                ${stats.recommendation}
            </div>
            <div class="text-center text-xs text-gray-400">
                Confiance: ${stats.confidence.toFixed(0)}%
            </div>
        </div>
    `;
}

function formatVolume(vol) {
    if (vol > 1e9) return (vol/1e9).toFixed(2) + 'B';
    if (vol > 1e6) return (vol/1e6).toFixed(2) + 'M';
    if (vol > 1e3) return (vol/1e3).toFixed(2) + 'K';
    return vol.toFixed(0);
}

// ========== INDICATORS SUMMARY (VERSION DEBUG) ==========

function updateIndicatorsSummary(data) {
    const signals = data.signals;
    const container = document.getElementById('indicators-summary');
    
    console.log('🔍 Signals reçus:', signals); // DEBUG
    
    if (!signals || typeof signals !== 'object') {
        container.innerHTML = '<div class="text-sm text-gray-400 text-center py-4">⚠️ Pas de signaux (signals manquant)</div>';
        return;
    }
    
    let html = '<div class="space-y-3 text-sm">';
    let hasAnySignal = false;
    
    // Trend
    if (signals.trend && typeof signals.trend === 'object' && Object.keys(signals.trend).length > 0) {
        html += '<div><div class="font-bold text-blue-400 mb-2">📈 Tendance</div>';
        for (const [key, val] of Object.entries(signals.trend)) {
            const signalText = val.signal || 'UNKNOWN';
            const color = signalText.includes('BUY') ? 'text-green-400' : 
                         signalText.includes('SELL') ? 'text-red-400' : 'text-gray-400';
            html += `<div class="flex justify-between text-xs mb-1">
                <span class="text-gray-400">${key.toUpperCase()}</span>
                <span class="${color} font-bold">${signalText}</span>
            </div>`;
            hasAnySignal = true;
        }
        html += '</div>';
    }
    
    // Momentum
    if (signals.momentum && typeof signals.momentum === 'object' && Object.keys(signals.momentum).length > 0) {
        html += '<div><div class="font-bold text-purple-400 mb-2">⚡ Momentum</div>';
        for (const [key, val] of Object.entries(signals.momentum)) {
            const signalText = val.signal || 'UNKNOWN';
            const color = signalText === 'OVERSOLD' ? 'text-green-400' : 
                         signalText === 'OVERBOUGHT' ? 'text-red-400' : 'text-gray-400';
            html += `<div class="flex justify-between text-xs mb-1">
                <span class="text-gray-400">${key.toUpperCase()}</span>
                <span class="${color} font-bold">${signalText}</span>
            </div>`;
            hasAnySignal = true;
        }
        html += '</div>';
    }
    
    // Volatility
    if (signals.volatility && typeof signals.volatility === 'object' && Object.keys(signals.volatility).length > 0) {
        html += '<div><div class="font-bold text-orange-400 mb-2">🌊 Volatilité</div>';
        for (const [key, val] of Object.entries(signals.volatility)) {
            const signalText = val.signal || 'UNKNOWN';
            const color = signalText === 'OVERSOLD' ? 'text-green-400' : 
                         signalText === 'OVERBOUGHT' ? 'text-red-400' : 'text-gray-400';
            html += `<div class="flex justify-between text-xs mb-1">
                <span class="text-gray-400">${key.toUpperCase()}</span>
                <span class="${color} font-bold">${signalText}</span>
            </div>`;
            hasAnySignal = true;
        }
        html += '</div>';
    }
    
    // Volume
    if (signals.volume && typeof signals.volume === 'object' && Object.keys(signals.volume).length > 0) {
        html += '<div><div class="font-bold text-cyan-400 mb-2">📊 Volume</div>';
        for (const [key, val] of Object.entries(signals.volume)) {
            const signalText = val.signal || 'UNKNOWN';
            const color = signalText === 'OVERSOLD' ? 'text-green-400' : 
                         signalText === 'OVERBOUGHT' ? 'text-red-400' : 'text-gray-400';
            html += `<div class="flex justify-between text-xs mb-1">
                <span class="text-gray-400">${key.toUpperCase()}</span>
                <span class="${color} font-bold">${signalText}</span>
            </div>`;
            hasAnySignal = true;
        }
        html += '</div>';
    }
    
    html += '</div>';
    
    if (!hasAnySignal) {
        console.warn('⚠️ Aucun signal trouvé dans:', signals);
        html = '<div class="text-sm text-gray-400 text-center py-4">❌ Aucun signal disponible</div>';
    }
    
    container.innerHTML = html;
}

// ========== AI ASSISTANT ==========

function generateAIAnalysis(data) {
    const stats = data.quick_stats;
    
    let message = `🎯 **${data.ticker}** est à **${stats.price.toFixed(2)}$** `;
    message += stats.change_pct >= 0 ? `(🟢 +${stats.change_pct.toFixed(2)}%)` : `(🔴 ${stats.change_pct.toFixed(2)}%)`;
    message += `\n\n**Signal**: ${stats.recommendation} (${stats.confidence.toFixed(0)}% confiance)\n\n`;
    
    if (stats.rsi > 70) {
        message += `⚠️ RSI en sur-achat (${stats.rsi.toFixed(1)}). `;
    } else if (stats.rsi < 30) {
        message += `✅ RSI en sur-vente (${stats.rsi.toFixed(1)}). `;
    }
    
    if (stats.adx > 25) {
        message += `Tendance forte (ADX ${stats.adx.toFixed(1)}).`;
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
    
    try {
        const res = await fetch(`${API_BASE}/api/ai-chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                ticker: currentData.ticker,
                context: currentData.quick_stats
            })
        });
        
        const data = await res.json();
        addAIMessage(data.response, 'ai');
        
    } catch (error) {
        addAIMessage(`❌ ${error.message}`, 'ai');
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

document.querySelectorAll('.indicator-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        const indicator = btn.dataset.indicator;
        
        if (activeIndicators.has(indicator)) {
            activeIndicators.delete(indicator);
            btn.classList.remove('active');
        } else {
            activeIndicators.add(indicator);
            btn.classList.add('active');
        }
        
        if (currentData) renderMainChart(currentData);
    });
});

window.addEventListener('load', () => {
    analyzeStock();
});
