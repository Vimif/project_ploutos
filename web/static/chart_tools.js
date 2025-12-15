/**
 * CHART TOOLS - Fibonacci, Volume Profile, Support/Resistance, Annotations
 * Gère l'affichage des outils graphiques avancés sur le chart principal
 */

let chartToolsState = {
    fibonacci: false,
    volumeProfile: false,
    supportResistance: false,
    annotations: false,
    zen: false
};

let chartToolsData = {
    fibonacci: null,
    volumeProfile: null,
    supportResistance: null
};

// ========== INIT CHART TOOLS ==========

function initChartTools() {
    // Fibonacci button
    const fibBtn = document.getElementById('fibonacci-toggle');
    if (fibBtn) {
        fibBtn.addEventListener('click', () => toggleChartTool('fibonacci'));
    }
    
    // Volume Profile button
    const vpBtn = document.getElementById('volume-profile-toggle');
    if (vpBtn) {
        vpBtn.addEventListener('click', () => toggleChartTool('volumeProfile'));
    }
    
    // Support/Resistance button
    const srBtn = document.getElementById('sr-toggle');
    if (srBtn) {
        srBtn.addEventListener('click', () => toggleChartTool('supportResistance'));
    }
    
    // Annotations button
    const annBtn = document.getElementById('annotations-toggle');
    if (annBtn) {
        annBtn.addEventListener('click', () => toggleChartTool('annotations'));
    }
    
    // Zen mode button
    const zenBtn = document.getElementById('zen-toggle');
    if (zenBtn) {
        zenBtn.addEventListener('click', () => toggleChartTool('zen'));
    }
    
    console.log('✅ Chart Tools initialisés');
}

// ========== TOGGLE TOOLS ==========

async function toggleChartTool(tool) {
    chartToolsState[tool] = !chartToolsState[tool];
    
    // Update button state
    const btnId = {
        'fibonacci': 'fibonacci-toggle',
        'volumeProfile': 'volume-profile-toggle',
        'supportResistance': 'sr-toggle',
        'annotations': 'annotations-toggle',
        'zen': 'zen-toggle'
    }[tool];
    
    const btn = document.getElementById(btnId);
    if (btn) {
        if (chartToolsState[tool]) {
            btn.classList.add('bg-green-600');
            btn.classList.remove('bg-gray-700');
        } else {
            btn.classList.remove('bg-green-600');
            btn.classList.add('bg-gray-700');
        }
    }
    
    // Load data if needed
    if (chartToolsState[tool] && !chartToolsData[tool] && currentData) {
        await loadChartToolData(tool);
    }
    
    // Redraw chart
    if (currentData) {
        renderMainChart(currentData);
    }
}

// ========== LOAD TOOL DATA ==========

async function loadChartToolData(tool) {
    if (!currentData || !currentData.ticker) return;
    
    const ticker = currentData.ticker;
    const period = document.getElementById('period-select')?.value || '3mo';
    
    try {
        if (tool === 'fibonacci') {
            const res = await fetch(`${API_BASE}/api/chart/${ticker}/fibonacci?period=${period}`);
            chartToolsData.fibonacci = await res.json();
            console.log('📐 Fibonacci chargé:', chartToolsData.fibonacci);
        }
        
        if (tool === 'volumeProfile') {
            const res = await fetch(`${API_BASE}/api/chart/${ticker}/volume-profile?period=${period}`);
            chartToolsData.volumeProfile = await res.json();
            console.log('📊 Volume Profile chargé:', chartToolsData.volumeProfile);
        }
        
        if (tool === 'supportResistance') {
            const res = await fetch(`${API_BASE}/api/chart/${ticker}/support-resistance?period=${period}`);
            chartToolsData.supportResistance = await res.json();
            console.log('🎯 S/R chargé:', chartToolsData.supportResistance);
        }
    } catch (error) {
        console.error(`❌ Erreur chargement ${tool}:`, error);
    }
}

// ========== ADD TOOLS TO CHART ==========

function addChartToolsToPlot(traces, data) {
    // 📐 FIBONACCI LEVELS
    if (chartToolsState.fibonacci && chartToolsData.fibonacci) {
        const fib = chartToolsData.fibonacci;
        if (fib.success && fib.retracements) {
            const colors = {
                '0.0%': '#ef4444',
                '23.6%': '#f97316',
                '38.2%': '#f59e0b',
                '50.0%': '#eab308',
                '61.8%': '#84cc16',
                '78.6%': '#22c55e',
                '100.0%': '#10b981'
            };
            
            Object.entries(fib.retracements).forEach(([level, price]) => {
                traces.push({
                    type: 'scatter',
                    mode: 'lines',
                    x: [data.dates[0], data.dates[data.dates.length - 1]],
                    y: [price, price],
                    name: `Fib ${level}`,
                    line: {
                        color: colors[level] || '#6366f1',
                        width: 1,
                        dash: 'dash'
                    },
                    hovertemplate: `${level}: $${price.toFixed(2)}<extra></extra>`
                });
            });
        }
    }
    
    // 🎯 SUPPORT & RESISTANCE
    if (chartToolsState.supportResistance && chartToolsData.supportResistance) {
        const sr = chartToolsData.supportResistance;
        if (sr.success) {
            // Resistance levels
            if (sr.resistance) {
                sr.resistance.forEach((r, idx) => {
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: [data.dates[0], data.dates[data.dates.length - 1]],
                        y: [r.level, r.level],
                        name: `R${idx + 1} (${r.strength})`,
                        line: {
                            color: '#ef4444',
                            width: 1 + (r.strength / 10),
                            dash: 'dot'
                        },
                        hovertemplate: `Résistance: $${r.level.toFixed(2)} (${r.strength} touches)<extra></extra>`
                    });
                });
            }
            
            // Support levels
            if (sr.support) {
                sr.support.forEach((s, idx) => {
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: [data.dates[0], data.dates[data.dates.length - 1]],
                        y: [s.level, s.level],
                        name: `S${idx + 1} (${s.strength})`,
                        line: {
                            color: '#10b981',
                            width: 1 + (s.strength / 10),
                            dash: 'dot'
                        },
                        hovertemplate: `Support: $${s.level.toFixed(2)} (${s.strength} touches)<extra></extra>`
                    });
                });
            }
        }
    }
    
    // 📊 VOLUME PROFILE (affichage lateral)
    if (chartToolsState.volumeProfile && chartToolsData.volumeProfile) {
        const vp = chartToolsData.volumeProfile;
        if (vp.success && vp.poc) {
            // Point of Control (POC)
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: [data.dates[0], data.dates[data.dates.length - 1]],
                y: [vp.poc.price, vp.poc.price],
                name: 'POC (Volume Max)',
                line: {
                    color: '#8b5cf6',
                    width: 2,
                    dash: 'solid'
                },
                hovertemplate: `POC: $${vp.poc.price.toFixed(2)}<extra></extra>`
            });
            
            // Value Area High/Low
            if (vp.value_area) {
                if (vp.value_area.high) {
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: [data.dates[0], data.dates[data.dates.length - 1]],
                        y: [vp.value_area.high, vp.value_area.high],
                        name: 'VAH (70%)',
                        line: {color: '#a855f7', width: 1, dash: 'dash'},
                        hovertemplate: `VAH: $${vp.value_area.high.toFixed(2)}<extra></extra>`
                    });
                }
                
                if (vp.value_area.low) {
                    traces.push({
                        type: 'scatter',
                        mode: 'lines',
                        x: [data.dates[0], data.dates[data.dates.length - 1]],
                        y: [vp.value_area.low, vp.value_area.low],
                        name: 'VAL (70%)',
                        line: {color: '#a855f7', width: 1, dash: 'dash'},
                        hovertemplate: `VAL: $${vp.value_area.low.toFixed(2)}<extra></extra>`
                    });
                }
            }
        }
    }
    
    return traces;
}

// ========== ZEN MODE ==========

function toggleZenMode() {
    const sidebar = document.querySelector('.sidebar');
    const indicators = document.querySelector('.indicators-panel');
    
    if (chartToolsState.zen) {
        // Hide sidebars
        if (sidebar) sidebar.style.display = 'none';
        if (indicators) indicators.style.display = 'none';
    } else {
        // Show sidebars
        if (sidebar) sidebar.style.display = 'block';
        if (indicators) indicators.style.display = 'block';
    }
}

// ========== ANNOTATIONS MODE ==========

function enableAnnotationsMode() {
    if (chartToolsState.annotations) {
        console.log('📋 Mode Annotations activé');
        // TODO: Implémenter mode dessin avec Plotly editable shapes
        alert('🚧 Mode Annotations en cours de développement');
    }
}

// ========== INTEGRATION AVEC ADVANCED_CHART.JS ==========

// Modifier renderMainChart pour intégrer les outils
const originalRenderMainChart = window.renderMainChart;

window.renderMainChart = function(data) {
    // Appeler la fonction originale
    originalRenderMainChart(data);
    
    // Récupérer le chart existant
    const chartEl = document.getElementById('main-chart');
    if (!chartEl) return;
    
    // Ajouter les outils au chart
    const currentTraces = chartEl.data || [];
    const newTraces = addChartToolsToPlot([...currentTraces], data);
    
    // Update le chart avec les nouveaux traces
    if (newTraces.length > currentTraces.length) {
        Plotly.addTraces('main-chart', newTraces.slice(currentTraces.length));
    }
};

// ========== AUTO-INIT ==========

window.addEventListener('load', () => {
    initChartTools();
    console.log('🛠️ Chart Tools prêt');
});

// Recharger les données tools quand le ticker change
window.addEventListener('chartDataLoaded', async (e) => {
    // Reset data
    chartToolsData = {
        fibonacci: null,
        volumeProfile: null,
        supportResistance: null
    };
    
    // Recharger si tools actifs
    for (const tool of Object.keys(chartToolsState)) {
        if (chartToolsState[tool] && tool !== 'annotations' && tool !== 'zen') {
            await loadChartToolData(tool);
        }
    }
    
    // Redraw chart avec les tools
    if (currentData) {
        renderMainChart(currentData);
    }
});
