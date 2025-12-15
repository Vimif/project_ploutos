/**
 * ⚡ SCALPER PRO - Trading Court-Terme Temps Réel
 * Day Trading & Swing Trading optimisé
 */

const API_BASE = window.location.origin;

let scalperState = {
    ticker: 'AAPL',
    timeframe: '1h',
    autoRefresh: true,
    soundAlerts: false,
    refreshInterval: null,
    lastSignal: null,
    data: null
};

// 🚀 INIT
window.addEventListener('load', () => {
    initScalper();
});

function initScalper() {
    console.log('⚡ Scalper Pro initializing...');
    
    // Event listeners
    document.getElementById('ticker-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeStock();
    });
    
    document.getElementById('analyze-btn').addEventListener('click', analyzeStock);
    
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('tf-active'));
            e.target.classList.add('tf-active');
            scalperState.timeframe = e.target.dataset.tf;
            analyzeStock();
        });
    });
    
    document.getElementById('auto-refresh').addEventListener('change', (e) => {
        scalperState.autoRefresh = e.target.checked;
        if (scalperState.autoRefresh) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });
    
    document.getElementById('sound-alerts').addEventListener('change', (e) => {
        scalperState.soundAlerts = e.target.checked;
    });
    
    document.getElementById('risk-amount').addEventListener('input', updatePositionSize);
    document.getElementById('risk-percent').addEventListener('input', updatePositionSize);
    
    // Auto-load
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('ticker')) {
        scalperState.ticker = urlParams.get('ticker').toUpperCase();
        document.getElementById('ticker-input').value = scalperState.ticker;
    }
    
    analyzeStock();
    startAutoRefresh();
}

function startAutoRefresh() {
    if (scalperState.refreshInterval) clearInterval(scalperState.refreshInterval);
    
    scalperState.refreshInterval = setInterval(() => {
        if (scalperState.autoRefresh) {
            console.log('🔄 Auto-refresh...');
            analyzeStock(true);
        }
    }, 30000); // 30 secondes
}

function stopAutoRefresh() {
    if (scalperState.refreshInterval) {
        clearInterval(scalperState.refreshInterval);
        scalperState.refreshInterval = null;
    }
}

async function analyzeStock(silent = false) {
    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
    if (!ticker) return;
    
    scalperState.ticker = ticker;
    
    if (!silent) {
        document.getElementById('refresh-icon').classList.add('fa-spin');
    }
    
    try {
        // Charger données chart
        const period = getPeriodFromTimeframe(scalperState.timeframe);
        const interval = getIntervalFromTimeframe(scalperState.timeframe);
        
        const [chartRes, proRes, srRes] = await Promise.all([
            fetch(`${API_BASE}/api/chart/${ticker}?period=${period}`),
            fetch(`${API_BASE}/api/pro-analysis/${ticker}`),
            fetch(`${API_BASE}/api/chart/${ticker}/support-resistance?period=${period}`)
        ]);
        
        const chartData = await chartRes.json();
        const proData = await proRes.json();
        const srData = await srRes.json();
        
        scalperState.data = { chart: chartData, pro: proData, sr: srData };
        
        // Update UI
        renderChart(chartData);
        updateSignals(proData);
        updateKeyLevels(srData);
        updateIndicators(chartData);
        updateHeatmap(chartData, srData);
        updateOrderFlow(chartData);
        updatePositionSize();
        
        // Check for signal change
        checkSignalChange(proData);
        
        document.getElementById('last-update').textContent = new Date().toLocaleTimeString('fr-FR');
        
    } catch (error) {
        console.error('❌ Error:', error);
    } finally {
        document.getElementById('refresh-icon').classList.remove('fa-spin');
    }
}

function getPeriodFromTimeframe(tf) {
    const map = { '5m': '1d', '15m': '5d', '1h': '1mo', '4h': '3mo', '1d': '6mo' };
    return map[tf] || '1mo';
}

function getIntervalFromTimeframe(tf) {
    const map = { '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d' };
    return map[tf] || '1h';
}

function renderChart(data) {
    const traces = [
        {
            type: 'candlestick',
            x: data.dates,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            increasing: { line: { color: '#00f260' } },
            decreasing: { line: { color: '#ff4757' } }
        }
    ];
    
    // SMA 20
    if (data.indicators && data.indicators.sma_20) {
        traces.push({
            type: 'scatter',
            x: data.dates,
            y: data.indicators.sma_20,
            name: 'SMA 20',
            line: { color: '#3b82f6', width: 1.5 }
        });
    }
    
    const layout = {
        plot_bgcolor: '#0a0e1a',
        paper_bgcolor: '#0a0e1a',
        font: { color: '#9ca3af', size: 10 },
        xaxis: { gridcolor: '#1f2937' },
        yaxis: { gridcolor: '#1f2937', side: 'right' },
        margin: { l: 10, r: 60, t: 10, b: 30 },
        showlegend: false
    };
    
    Plotly.newPlot('main-chart', traces, layout, { responsive: true, displayModeBar: false });
}

function updateSignals(proData) {
    if (!proData || !proData.overall_signal) {
        document.getElementById('main-signal').textContent = '?';
        document.getElementById('main-signal-text').textContent = 'ATTENTE';
        return;
    }
    
    const signal = proData.overall_signal;
    const signalText = signal.recommendation || 'HOLD';
    const signalIcon = { 'STRONG BUY': '🚀', 'BUY': '👍', 'HOLD': '✋', 'SELL': '👎', 'STRONG SELL': '⚠️' };
    
    document.getElementById('main-signal').textContent = signalIcon[signalText] || '✋';
    document.getElementById('main-signal-text').textContent = signalText;
    
    const badge = document.getElementById('main-signal-text');
    badge.className = 'px-6 py-3 rounded-xl font-bold text-xl mb-3';
    
    if (signalText.includes('BUY')) {
        badge.classList.add('signal-buy');
    } else if (signalText.includes('SELL')) {
        badge.classList.add('signal-sell');
    } else {
        badge.classList.add('signal-neutral');
    }
    
    // Update entry/stop/target
    const price = scalperState.data.chart.current_price || 0;
    document.getElementById('entry-price').textContent = `$${price.toFixed(2)}`;
    
    const stopLoss = signal.stop_loss || (price * 0.98);
    const takeProfit = signal.take_profit || (price * 1.03);
    
    document.getElementById('stop-loss').textContent = `$${stopLoss.toFixed(2)}`;
    document.getElementById('take-profit').textContent = `$${takeProfit.toFixed(2)}`;
    
    const risk = Math.abs(price - stopLoss);
    const reward = Math.abs(takeProfit - price);
    const rrRatio = risk > 0 ? (reward / risk).toFixed(2) : '-';
    
    document.getElementById('rr-ratio').textContent = `1:${rrRatio}`;
    
    // Update TF badge
    document.getElementById('tf-signal-badge').textContent = signalText;
    document.getElementById('tf-signal-badge').className = 'px-4 py-1 rounded-lg font-bold text-sm';
    if (signalText.includes('BUY')) {
        document.getElementById('tf-signal-badge').classList.add('signal-buy');
    } else if (signalText.includes('SELL')) {
        document.getElementById('tf-signal-badge').classList.add('signal-sell');
    } else {
        document.getElementById('tf-signal-badge').classList.add('signal-neutral');
    }
}

function updateKeyLevels(srData) {
    const container = document.getElementById('key-levels');
    if (!srData || !srData.success) {
        container.innerHTML = '<div class="text-gray-500 text-center py-2">Aucune donnée</div>';
        return;
    }
    
    let html = '';
    
    // Resistances
    if (srData.resistance && srData.resistance.length > 0) {
        srData.resistance.slice(0, 5).forEach((r, idx) => {
            const price = r.level || r.price || r;
            const strength = r.strength || 1;
            html += `
                <div class="flex items-center gap-2">
                    <div class="level-line level-resistance" style="width: ${strength * 20}px"></div>
                    <span class="text-red-400 font-mono">R${idx + 1}: $${price.toFixed(2)}</span>
                </div>
            `;
        });
    }
    
    // Current price
    if (srData.current_price) {
        html += `
            <div class="flex items-center gap-2 my-2">
                <div class="level-line level-poc"></div>
                <span class="text-purple-400 font-mono font-bold">NOW: $${srData.current_price.toFixed(2)}</span>
            </div>
        `;
    }
    
    // Supports
    if (srData.support && srData.support.length > 0) {
        srData.support.slice(0, 5).forEach((s, idx) => {
            const price = s.level || s.price || s;
            const strength = s.strength || 1;
            html += `
                <div class="flex items-center gap-2">
                    <div class="level-line level-support" style="width: ${strength * 20}px"></div>
                    <span class="text-green-400 font-mono">S${idx + 1}: $${price.toFixed(2)}</span>
                </div>
            `;
        });
    }
    
    container.innerHTML = html || '<div class="text-gray-500">Aucun niveau détecté</div>';
}

function updateIndicators(data) {
    const indicators = data.indicators || {};
    
    // RSI
    if (indicators.rsi && indicators.rsi.length > 0) {
        const rsi = indicators.rsi[indicators.rsi.length - 1];
        document.getElementById('rsi-value').textContent = rsi.toFixed(1);
        document.getElementById('rsi-value').style.color = rsi > 70 ? '#ef4444' : rsi < 30 ? '#10b981' : '#fbbf24';
    }
    
    // MACD
    if (indicators.macd && indicators.macd_signal) {
        const macd = indicators.macd[indicators.macd.length - 1];
        const signal = indicators.macd_signal[indicators.macd_signal.length - 1];
        const macdSignal = macd > signal ? '↑ BUY' : '↓ SELL';
        document.getElementById('macd-signal').textContent = macdSignal;
        document.getElementById('macd-signal').style.color = macd > signal ? '#10b981' : '#ef4444';
    }
    
    // ADX
    if (indicators.adx && indicators.adx.length > 0) {
        const adx = indicators.adx[indicators.adx.length - 1];
        document.getElementById('adx-value').textContent = adx.toFixed(1);
        document.getElementById('adx-value').style.color = adx > 25 ? '#10b981' : '#6b7280';
    }
    
    // Volume change
    const volumeChange = data.volume_24h ? '+' + formatVolume(data.volume_24h) : '-';
    document.getElementById('volume-change').textContent = volumeChange;
}

function updateHeatmap(chartData, srData) {
    const container = document.getElementById('price-heatmap');
    const currentPrice = chartData.current_price || 0;
    const range = currentPrice * 0.1; // +/- 10%
    const min = currentPrice - range;
    const max = currentPrice + range;
    const step = (max - min) / 50;
    
    let html = '';
    for (let i = 0; i < 50; i++) {
        const price = min + (i * step);
        
        // Calculer intensité (distance aux supports/résistances)
        let intensity = 50; // Neutre
        
        if (srData && srData.success) {
            // Check proximity to resistance
            (srData.resistance || []).forEach(r => {
                const rPrice = r.level || r.price || r;
                const dist = Math.abs(price - rPrice);
                if (dist < step * 2) {
                    intensity = Math.min(100, intensity + 30);
                }
            });
            
            // Check proximity to support
            (srData.support || []).forEach(s => {
                const sPrice = s.level || s.price || s;
                const dist = Math.abs(price - sPrice);
                if (dist < step * 2) {
                    intensity = Math.max(0, intensity - 30);
                }
            });
        }
        
        const color = `hsl(${120 - intensity * 1.2}, 70%, ${40 + intensity * 0.2}%)`;
        html += `<div class="heatmap-cell" style="background: ${color}; height: 30px; border-radius: 2px;" title="$${price.toFixed(2)}"></div>`;
    }
    
    container.innerHTML = html;
}

function updateOrderFlow(data) {
    // Simuler order flow basé sur le price action
    const closes = data.close || [];
    if (closes.length < 2) return;
    
    const recent = closes.slice(-10);
    const bullishBars = recent.filter((c, i) => i > 0 && c > recent[i - 1]).length;
    const bearishBars = recent.length - 1 - bullishBars;
    
    const buyPressure = (bullishBars / (recent.length - 1)) * 100;
    const sellPressure = (bearishBars / (recent.length - 1)) * 100;
    
    document.getElementById('buy-pressure-pct').textContent = `${buyPressure.toFixed(0)}%`;
    document.getElementById('sell-pressure-pct').textContent = `${sellPressure.toFixed(0)}%`;
    
    document.getElementById('buy-pressure-bar').style.width = `${buyPressure}%`;
    document.getElementById('sell-pressure-bar').style.width = `${sellPressure}%`;
    
    const ratio = buyPressure > 0 ? (buyPressure / sellPressure).toFixed(2) : '0.00';
    document.getElementById('volume-ratio').textContent = ratio;
    document.getElementById('volume-ratio').style.color = buyPressure > sellPressure ? '#10b981' : '#ef4444';
}

function updatePositionSize() {
    if (!scalperState.data || !scalperState.data.chart) return;
    
    const capital = parseFloat(document.getElementById('risk-amount').value) || 0;
    const riskPct = parseFloat(document.getElementById('risk-percent').value) || 0;
    const currentPrice = scalperState.data.chart.current_price || 0;
    
    const entryPrice = currentPrice;
    const stopLoss = parseFloat(document.getElementById('stop-loss').textContent.replace('$', '')) || (currentPrice * 0.98);
    const takeProfit = parseFloat(document.getElementById('take-profit').textContent.replace('$', '')) || (currentPrice * 1.03);
    
    const riskPerShare = Math.abs(entryPrice - stopLoss);
    const riskAmount = capital * (riskPct / 100);
    
    const shares = riskPerShare > 0 ? Math.floor(riskAmount / riskPerShare) : 0;
    const positionValue = shares * entryPrice;
    
    const maxLoss = shares * riskPerShare;
    const potentialProfit = shares * Math.abs(takeProfit - entryPrice);
    
    document.getElementById('position-size').textContent = `${shares} actions ($${positionValue.toFixed(2)})`;
    document.getElementById('max-loss').textContent = `$${maxLoss.toFixed(2)}`;
    document.getElementById('potential-profit').textContent = `$${potentialProfit.toFixed(2)}`;
}

function checkSignalChange(proData) {
    if (!proData || !proData.overall_signal) return;
    
    const newSignal = proData.overall_signal.recommendation;
    
    if (scalperState.lastSignal && scalperState.lastSignal !== newSignal) {
        console.log(`🚨 Signal changed: ${scalperState.lastSignal} → ${newSignal}`);
        
        if (scalperState.soundAlerts) {
            playAlert();
        }
        
        // Show notification if supported
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('Ploutos Scalper', {
                body: `${scalperState.ticker}: ${newSignal}`,
                icon: '/static/logo.png'
            });
        }
    }
    
    scalperState.lastSignal = newSignal;
}

function playAlert() {
    const audio = document.getElementById('alert-sound');
    if (audio) {
        audio.play().catch(e => console.warn('Audio play failed:', e));
    }
}

function formatVolume(vol) {
    if (vol > 1e9) return (vol / 1e9).toFixed(2) + 'B';
    if (vol > 1e6) return (vol / 1e6).toFixed(2) + 'M';
    if (vol > 1e3) return (vol / 1e3).toFixed(2) + 'K';
    return vol.toFixed(0);
}

console.log('⚡ Scalper Pro loaded');
