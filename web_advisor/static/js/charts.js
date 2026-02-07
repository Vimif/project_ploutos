/**
 * Ploutos Advisory - Constructeurs de graphiques Plotly.js
 *
 * Fonctions pour creer les graphiques en chandelier, indicateurs, previsions et jauges.
 */

const DARK_LAYOUT = {
    paper_bgcolor: '#1e293b',
    plot_bgcolor: '#1e293b',
    font: { color: '#94a3b8', size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 30 },
    xaxis: {
        gridcolor: '#334155',
        linecolor: '#475569',
        rangeslider: { visible: false },
    },
    yaxis: {
        gridcolor: '#334155',
        linecolor: '#475569',
    },
    legend: {
        bgcolor: 'transparent',
        font: { size: 10, color: '#94a3b8' },
        orientation: 'h',
        y: 1.02,
    },
    hovermode: 'x unified',
};

const PLOTLY_CONFIG = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
    displaylogo: false,
    responsive: true,
};


/**
 * Graphique en chandelier avec overlays (SMA, Bollinger)
 */
function buildCandlestickChart(containerId, ohlcv, indicators, analysis) {
    if (!ohlcv || ohlcv.length === 0) return;

    const dates = ohlcv.map(d => d.date);
    const traces = [];

    // Chandelier
    traces.push({
        type: 'candlestick',
        x: dates,
        open: ohlcv.map(d => d.open),
        high: ohlcv.map(d => d.high),
        low: ohlcv.map(d => d.low),
        close: ohlcv.map(d => d.close),
        increasing: { line: { color: '#10b981' }, fillcolor: '#10b981' },
        decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
        name: 'Prix',
        yaxis: 'y',
    });

    // SMA overlays
    if (indicators.sma_20) {
        traces.push({
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.sma_20,
            line: { color: '#60a5fa', width: 1 },
            name: 'SMA 20',
            yaxis: 'y',
        });
    }
    if (indicators.sma_50) {
        traces.push({
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.sma_50,
            line: { color: '#f59e0b', width: 1 },
            name: 'SMA 50',
            yaxis: 'y',
        });
    }

    // Bollinger Bands
    if (indicators.bb_upper && indicators.bb_lower) {
        traces.push({
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.bb_upper,
            line: { color: '#64748b', width: 1, dash: 'dot' },
            name: 'BB Sup',
            yaxis: 'y',
        });
        traces.push({
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.bb_lower,
            line: { color: '#64748b', width: 1, dash: 'dot' },
            fill: 'tonexty',
            fillcolor: 'rgba(100,116,139,0.05)',
            name: 'BB Inf',
            yaxis: 'y',
        });
    }

    // Volume (subplot)
    traces.push({
        type: 'bar',
        x: dates,
        y: ohlcv.map(d => d.volume),
        marker: {
            color: ohlcv.map((d, i) => {
                if (i === 0) return '#64748b';
                return d.close >= ohlcv[i-1].close ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)';
            }),
        },
        name: 'Volume',
        yaxis: 'y2',
    });

    // Lignes entry/stop/take-profit
    const shapes = [];
    if (analysis.entry_price) {
        shapes.push({
            type: 'line', xref: 'paper', x0: 0, x1: 1,
            y0: analysis.entry_price, y1: analysis.entry_price,
            line: { color: '#60a5fa', width: 1, dash: 'dash' },
        });
    }
    if (analysis.stop_loss) {
        shapes.push({
            type: 'line', xref: 'paper', x0: 0, x1: 1,
            y0: analysis.stop_loss, y1: analysis.stop_loss,
            line: { color: '#ef4444', width: 1, dash: 'dash' },
        });
    }
    if (analysis.take_profit) {
        shapes.push({
            type: 'line', xref: 'paper', x0: 0, x1: 1,
            y0: analysis.take_profit, y1: analysis.take_profit,
            line: { color: '#10b981', width: 1, dash: 'dash' },
        });
    }

    const layout = {
        ...DARK_LAYOUT,
        shapes: shapes,
        yaxis: {
            ...DARK_LAYOUT.yaxis,
            domain: [0.2, 1],
            title: 'Prix ($)',
        },
        yaxis2: {
            gridcolor: '#334155',
            domain: [0, 0.15],
            title: 'Vol',
            titlefont: { size: 9 },
        },
        xaxis: {
            ...DARK_LAYOUT.xaxis,
            type: 'date',
        },
    };

    Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}


/**
 * Graphique RSI
 */
function buildRSIChart(containerId, indicators) {
    if (!indicators.rsi) return;

    const traces = [
        {
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.rsi,
            line: { color: '#a78bfa', width: 1.5 },
            name: 'RSI',
            fill: 'tozeroy',
            fillcolor: 'rgba(167,139,250,0.05)',
        },
    ];

    const layout = {
        ...DARK_LAYOUT,
        margin: { l: 40, r: 10, t: 5, b: 25 },
        yaxis: {
            ...DARK_LAYOUT.yaxis,
            range: [0, 100],
        },
        shapes: [
            { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 70, line: { color: '#ef4444', width: 1, dash: 'dot' } },
            { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: { color: '#10b981', width: 1, dash: 'dot' } },
        ],
    };

    Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}


/**
 * Graphique MACD
 */
function buildMACDChart(containerId, indicators) {
    if (!indicators.macd_line) return;

    const traces = [
        {
            type: 'bar',
            x: indicators.dates,
            y: indicators.macd_histogram,
            marker: {
                color: indicators.macd_histogram.map(v =>
                    v === null ? '#64748b' : v >= 0 ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)'
                ),
            },
            name: 'Histogramme',
        },
        {
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.macd_line,
            line: { color: '#60a5fa', width: 1.5 },
            name: 'MACD',
        },
        {
            type: 'scatter', mode: 'lines',
            x: indicators.dates, y: indicators.macd_signal,
            line: { color: '#f59e0b', width: 1.5 },
            name: 'Signal',
        },
    ];

    const layout = {
        ...DARK_LAYOUT,
        margin: { l: 40, r: 10, t: 5, b: 25 },
    };

    Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}


/**
 * Graphique de prevision avec bandes de confiance
 */
function buildForecastChart(containerId, currentPrice, forecast) {
    if (!forecast || forecast.length === 0) return;

    const dates = forecast.map(f => f.date);
    const predicted = forecast.map(f => f.predicted);
    const lower80 = forecast.map(f => f.lower_80);
    const upper80 = forecast.map(f => f.upper_80);
    const lower95 = forecast.map(f => f.lower_95);
    const upper95 = forecast.map(f => f.upper_95);

    const traces = [
        // Bande 95%
        {
            type: 'scatter', mode: 'lines',
            x: dates, y: upper95,
            line: { width: 0 },
            showlegend: false,
        },
        {
            type: 'scatter', mode: 'lines',
            x: dates, y: lower95,
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(96,165,250,0.08)',
            name: 'IC 95%',
        },
        // Bande 80%
        {
            type: 'scatter', mode: 'lines',
            x: dates, y: upper80,
            line: { width: 0 },
            showlegend: false,
        },
        {
            type: 'scatter', mode: 'lines',
            x: dates, y: lower80,
            line: { width: 0 },
            fill: 'tonexty',
            fillcolor: 'rgba(96,165,250,0.15)',
            name: 'IC 80%',
        },
        // Ligne de prevision
        {
            type: 'scatter', mode: 'lines+markers',
            x: dates, y: predicted,
            line: { color: '#60a5fa', width: 2 },
            marker: { size: 5 },
            name: 'Prevision',
        },
        // Ligne prix actuel
        {
            type: 'scatter', mode: 'lines',
            x: [dates[0], dates[dates.length - 1]],
            y: [currentPrice, currentPrice],
            line: { color: '#94a3b8', width: 1, dash: 'dash' },
            name: 'Prix actuel',
        },
    ];

    const layout = {
        ...DARK_LAYOUT,
        yaxis: { ...DARK_LAYOUT.yaxis, title: 'Prix ($)' },
    };

    Plotly.newPlot(containerId, traces, layout, PLOTLY_CONFIG);
}


/**
 * Jauge de score composite
 */
function buildGaugeChart(containerId, score, recommendation) {
    // Convertir score [-1, +1] en valeur 0-100
    const value = Math.round((score + 1) / 2 * 100);

    const data = [{
        type: 'indicator',
        mode: 'gauge+number',
        value: value,
        number: { suffix: '%', font: { size: 18, color: '#e2e8f0' } },
        gauge: {
            axis: {
                range: [0, 100],
                tickwidth: 0,
                tickcolor: 'transparent',
                dtick: 25,
                tickfont: { size: 8, color: '#64748b' },
            },
            bar: { color: '#60a5fa', thickness: 0.6 },
            bgcolor: '#334155',
            borderwidth: 0,
            steps: [
                { range: [0, 20], color: '#7f1d1d' },
                { range: [20, 40], color: '#78350f' },
                { range: [40, 60], color: '#374151' },
                { range: [60, 80], color: '#064e3b' },
                { range: [80, 100], color: '#022c22' },
            ],
        },
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        margin: { l: 10, r: 10, t: 10, b: 0 },
        font: { color: '#94a3b8' },
    };

    Plotly.newPlot(containerId, data, layout, { displayModeBar: false, responsive: true });
}
