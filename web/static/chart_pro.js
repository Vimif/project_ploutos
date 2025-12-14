/**
 * 📈 CHART PRO - Frontend Avancé
 * 
 * Features:
 * - Fibonacci Retracement overlay
 * - Volume Profile sidebar
 * - Support/Resistance lines
 * - Smart Annotations (AI-powered)
 * - Keyboard shortcuts
 * - Multi-timeframe selector
 * - Drawing tools
 * - Zen mode
 */

class ChartPro {
    constructor() {
        this.currentTicker = this.getCurrentTicker();
        this.currentPeriod = '3mo';
        this.zenMode = false;
        this.showFibonacci = false;
        this.showVolumeProfile = false;
        this.showSupportResistance = true;
        this.showAnnotations = true;
        
        this.fibonacciData = null;
        this.volumeProfileData = null;
        this.supportResistanceData = null;
        this.annotationsData = null;
        
        this.initKeyboardShortcuts();
        this.createControlPanel();
        this.watchTickerChanges();
        
        // Auto-load S/R au démarrage
        setTimeout(() => {
            if (this.showSupportResistance) {
                this.loadSupportResistance();
            }
        }, 2000);
    }
    
    getCurrentTicker() {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('ticker')) return urlParams.get('ticker').toUpperCase();
        
        const tickerInput = document.getElementById('ticker-input');
        if (tickerInput && tickerInput.value) return tickerInput.value.toUpperCase();
        
        if (window.currentTicker) return window.currentTicker.toUpperCase();
        if (window.currentData && window.currentData.ticker) return window.currentData.ticker.toUpperCase();
        
        return 'AAPL';
    }
    
    watchTickerChanges() {
        let lastTicker = this.currentTicker;
        
        setInterval(() => {
            const newTicker = this.getCurrentTicker();
            if (newTicker !== lastTicker) {
                console.log(`🔄 Chart Pro - Ticker changé: ${lastTicker} → ${newTicker}`);
                lastTicker = newTicker;
                this.currentTicker = newTicker;
                this.onTickerChange();
            }
        }, 500);
    }
    
    onTickerChange() {
        this.fibonacciData = null;
        this.volumeProfileData = null;
        this.supportResistanceData = null;
        this.annotationsData = null;
        
        if (this.showFibonacci) this.loadFibonacci();
        if (this.showVolumeProfile) this.loadVolumeProfile();
        if (this.showSupportResistance) this.loadSupportResistance();
    }
    
    initKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key.toLowerCase()) {
                case 'z': this.toggleZenMode(); break;
                case 'f': this.toggleFibonacci(); break;
                case 'v': this.toggleVolumeProfile(); break;
                case 's': this.toggleSupportResistance(); break;
                case 'a': this.toggleAnnotations(); break;
                case '1': this.changePeriod('1d'); break;
                case '5': this.changePeriod('5d'); break;
                case 'm': this.changePeriod('1mo'); break;
                case 'y': this.changePeriod('1y'); break;
            }
        });
    }
    
    createControlPanel() {
        const panel = document.createElement('div');
        panel.id = 'chart-pro-controls';
        panel.innerHTML = `
            <div class="control-group">
                <button id="btn-fibonacci" class="control-btn" title="Toggle Fibonacci (F)">
                    📊 Fibonacci
                </button>
                <button id="btn-volume-profile" class="control-btn" title="Toggle Volume Profile (V)">
                    📉 Volume Profile
                </button>
                <button id="btn-support-resistance" class="control-btn active" title="Toggle S/R (S)">
                    🎯 Support/Resistance
                </button>
                <button id="btn-annotations" class="control-btn active" title="Toggle Annotations (A)">
                    ⭐ Annotations
                </button>
                <button id="btn-zen-mode" class="control-btn" title="Zen Mode (Z)">
                    🧘 Zen
                </button>
            </div>
            
            <div class="control-group timeframe-selector">
                <button class="tf-btn" data-period="1d">1D</button>
                <button class="tf-btn" data-period="5d">5D</button>
                <button class="tf-btn" data-period="1mo">1M</button>
                <button class="tf-btn active" data-period="3mo">3M</button>
                <button class="tf-btn" data-period="6mo">6M</button>
                <button class="tf-btn" data-period="1y">1Y</button>
                <button class="tf-btn" data-period="max">MAX</button>
            </div>
        `;
        
        const chartContainer = document.querySelector('.chart-container') || document.body;
        chartContainer.insertBefore(panel, chartContainer.firstChild);
        
        document.getElementById('btn-fibonacci').addEventListener('click', () => this.toggleFibonacci());
        document.getElementById('btn-volume-profile').addEventListener('click', () => this.toggleVolumeProfile());
        document.getElementById('btn-support-resistance').addEventListener('click', () => this.toggleSupportResistance());
        document.getElementById('btn-annotations').addEventListener('click', () => this.toggleAnnotations());
        document.getElementById('btn-zen-mode').addEventListener('click', () => this.toggleZenMode());
        
        document.querySelectorAll('.tf-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.changePeriod(e.target.dataset.period);
            });
        });
    }
    
    async toggleFibonacci() {
        this.showFibonacci = !this.showFibonacci;
        document.getElementById('btn-fibonacci').classList.toggle('active');
        
        if (this.showFibonacci && !this.fibonacciData) {
            await this.loadFibonacci();
        } else {
            this.updateChart();
        }
    }
    
    async toggleVolumeProfile() {
        this.showVolumeProfile = !this.showVolumeProfile;
        document.getElementById('btn-volume-profile').classList.toggle('active');
        
        if (this.showVolumeProfile && !this.volumeProfileData) {
            await this.loadVolumeProfile();
        } else {
            this.renderVolumeProfile();
        }
    }
    
    async toggleSupportResistance() {
        this.showSupportResistance = !this.showSupportResistance;
        document.getElementById('btn-support-resistance').classList.toggle('active');
        
        if (this.showSupportResistance && !this.supportResistanceData) {
            await this.loadSupportResistance();
        } else {
            this.updateChart();
        }
    }
    
    toggleAnnotations() {
        this.showAnnotations = !this.showAnnotations;
        document.getElementById('btn-annotations').classList.toggle('active');
        this.updateChart();
    }
    
    toggleZenMode() {
        this.zenMode = !this.zenMode;
        document.getElementById('btn-zen-mode').classList.toggle('active');
        document.body.classList.toggle('zen-mode');
        
        const hideElements = ['.header', '.nav', '.sidebar', '.footer'];
        hideElements.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) el.style.display = this.zenMode ? 'none' : '';
        });
    }
    
    changePeriod(period) {
        this.currentPeriod = period;
        this.loadChartData();
    }
    
    async loadFibonacci() {
        try {
            console.log(`🔄 Loading Fibonacci for ${this.currentTicker}...`);
            const response = await fetch(`/api/chart/${this.currentTicker}/fibonacci?period=${this.currentPeriod}`);
            this.fibonacciData = await response.json();
            console.log('✅ Fibonacci loaded:', this.fibonacciData);
            this.updateChart();
        } catch (error) {
            console.error('❌ Erreur Fibonacci:', error);
        }
    }
    
    async loadVolumeProfile() {
        try {
            console.log(`🔄 Loading Volume Profile for ${this.currentTicker}...`);
            const response = await fetch(`/api/chart/${this.currentTicker}/volume-profile?period=${this.currentPeriod}`);
            this.volumeProfileData = await response.json();
            console.log('✅ Volume Profile loaded:', this.volumeProfileData);
            this.renderVolumeProfile();
        } catch (error) {
            console.error('❌ Erreur Volume Profile:', error);
        }
    }
    
    async loadSupportResistance() {
        try {
            console.log(`🔄 Loading Support/Resistance for ${this.currentTicker}...`);
            const response = await fetch(`/api/chart/${this.currentTicker}/support-resistance?period=${this.currentPeriod}`);
            this.supportResistanceData = await response.json();
            console.log('✅ Support/Resistance loaded:', this.supportResistanceData);
            this.updateChart();
        } catch (error) {
            console.error('❌ Erreur S/R:', error);
        }
    }
    
    async loadChartData() {
        try {
            const response = await fetch(`/api/chart/${this.currentTicker}?period=${this.currentPeriod}`);
            const data = await response.json();
            
            this.fibonacciData = data.fibonacci;
            this.volumeProfileData = data.volume_profile;
            this.supportResistanceData = data.support_resistance;
            this.annotationsData = data.annotations;
            
            this.updateChart();
            this.renderVolumeProfile();
        } catch (error) {
            console.error('❌ Erreur chargement chart:', error);
        }
    }
    
    /**
     * 📊 Mettre à jour le chart (Plotly)
     */
    updateChart() {
        const chartElement = document.getElementById('main-chart');
        
        if (!chartElement) {
            console.warn('⚠️ main-chart element non trouvé');
            return;
        }
        
        if (!window.Plotly) {
            console.warn('⚠️ Plotly non chargé');
            return;
        }
        
        const shapes = [];
        const annotations = [];
        
        // 📊 Fibonacci Levels - 🔥 FILTRE les niveaux dans la zone visible
        if (this.showFibonacci && this.fibonacciData && this.fibonacciData.levels) {
            const fib = this.fibonacciData;
            const colors = {
                '0.0': '#ff4757',
                '23.6': '#ffa502',
                '38.2': '#fffa65',
                '50.0': '#3742fa',
                '61.8': '#2ed573',
                '78.6': '#1e90ff',
                '100.0': '#ff4757',
                '161.8': '#a29bfe',
                '261.8': '#fd79a8'
            };
            
            // 🔥 Déterminer la plage visible du chart
            const visibleMin = fib.swing_low * 0.95;  // 5% marge basse
            const visibleMax = fib.swing_high * 1.05;  // 5% marge haute
            
            let fibCount = 0;
            Object.entries(fib.levels).forEach(([level, price]) => {
                // 🔥 FILTRE: ignorer les niveaux hors zone visible
                if (price < visibleMin || price > visibleMax) {
                    console.log(`⚠️ Fibonacci ${level}% ($${price.toFixed(2)}) hors zone visible - ignoré`);
                    return;
                }
                
                shapes.push({
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    y0: price,
                    y1: price,
                    line: {
                        color: colors[level] || '#95a5a6',
                        width: level === '50.0' ? 2 : 1,
                        dash: 'dot'
                    }
                });
                
                annotations.push({
                    x: 0.98,  // 🔥 Position à 98% pour éviter le débordement
                    xref: 'paper',
                    y: price,
                    xanchor: 'right',
                    text: `Fib ${level}% $${price.toFixed(2)}`,
                    showarrow: false,
                    font: {
                        size: 9,
                        color: colors[level] || '#95a5a6'
                    },
                    bgcolor: 'rgba(0,0,0,0.7)',
                    borderpad: 2
                });
                
                fibCount++;
            });
            
            console.log(`✅ ${fibCount} Fibonacci levels added (visible range: $${visibleMin.toFixed(2)} - $${visibleMax.toFixed(2)})`);
        }
        
        // 🎯 Support/Resistance
        if (this.showSupportResistance && this.supportResistanceData) {
            // Supports (vert)
            (this.supportResistanceData.supports || []).forEach(support => {
                shapes.push({
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    y0: support.price,
                    y1: support.price,
                    line: {
                        color: '#00f260',
                        width: Math.min(support.strength || 2, 3),
                        dash: 'dash'
                    }
                });
                
                annotations.push({
                    x: 0.02,  // 🔥 Position à 2% à gauche
                    xref: 'paper',
                    y: support.price,
                    xanchor: 'left',
                    text: `Supp $${support.price.toFixed(2)}`,
                    showarrow: false,
                    font: { size: 9, color: '#00f260' },
                    bgcolor: 'rgba(0,242,96,0.2)',
                    borderpad: 2
                });
            });
            
            // Resistances (rouge)
            (this.supportResistanceData.resistances || []).forEach(resistance => {
                shapes.push({
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    y0: resistance.price,
                    y1: resistance.price,
                    line: {
                        color: '#ff4757',
                        width: Math.min(resistance.strength || 2, 3),
                        dash: 'dash'
                    }
                });
                
                annotations.push({
                    x: 0.02,
                    xref: 'paper',
                    y: resistance.price,
                    xanchor: 'left',
                    text: `Resi $${resistance.price.toFixed(2)}`,
                    showarrow: false,
                    font: { size: 9, color: '#ff4757' },
                    bgcolor: 'rgba(255,71,87,0.2)',
                    borderpad: 2
                });
            });
            
            const totalSR = (this.supportResistanceData.supports?.length || 0) + (this.supportResistanceData.resistances?.length || 0);
            console.log(`✅ ${totalSR} Support/Resistance levels added`);
        }
        
        // Update Plotly chart
        try {
            Plotly.relayout('main-chart', {
                shapes: shapes,
                annotations: annotations
            });
            console.log(`✅ Chart updated with ${shapes.length} shapes and ${annotations.length} annotations`);
        } catch (error) {
            console.error('❌ Erreur Plotly.relayout:', error);
        }
    }
    
    renderVolumeProfile() {
        let container = document.getElementById('volume-profile-container');
        
        if (!container) {
            container = document.createElement('div');
            container.id = 'volume-profile-container';
            container.className = 'volume-profile-sidebar';
            document.body.appendChild(container);
        }
        
        if (!this.showVolumeProfile) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        
        if (!this.volumeProfileData || !this.volumeProfileData.profile) {
            container.innerHTML = '<div class="loading">⌛ Chargement...</div>';
            return;
        }
        
        const vp = this.volumeProfileData;
        const maxVolume = Math.max(...vp.profile.map(p => p.volume));
        
        let html = `
            <h3>📉 Volume Profile</h3>
            <div class="vp-stats">
                <div>POC: <strong>$${vp.poc_price.toFixed(2)}</strong></div>
                <div>VA: $${vp.value_area_low.toFixed(2)} - $${vp.value_area_high.toFixed(2)}</div>
            </div>
            <div class="vp-bars">
        `;
        
        vp.profile.slice().reverse().forEach(level => {
            const widthPct = (level.volume / maxVolume * 100).toFixed(1);
            const isPoc = Math.abs(level.price - vp.poc_price) < 0.01;
            const inVa = level.price >= vp.value_area_low && level.price <= vp.value_area_high;
            
            const barClass = isPoc ? 'poc' : (inVa ? 'value-area' : '');
            
            html += `
                <div class="vp-row">
                    <div class="vp-price">$${level.price.toFixed(2)}</div>
                    <div class="vp-bar-container">
                        <div class="vp-bar ${barClass}" style="width: ${widthPct}%"></div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
        console.log('✅ Volume Profile rendered');
    }
}

// 🚀 Auto-init
let chartPro;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        chartPro = new ChartPro();
        console.log('✅ Chart Pro initialized');
    });
} else {
    chartPro = new ChartPro();
    console.log('✅ Chart Pro initialized');
}

// Export pour utilisation externe
window.ChartPro = ChartPro;
window.chartPro = chartPro;
