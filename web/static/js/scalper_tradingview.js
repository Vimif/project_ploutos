/**
 * ⚡ SCALPER PRO TRADINGVIEW
 * TradingView Advanced Charts + WebSocket Real-Time
 */

class ScalperTradingView {
    constructor() {
        this.currentTicker = 'AAPL';
        this.currentTimeframe = '5';
        this.widget = null;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.indicators = {};
        
        this.init();
    }

    init() {
        console.log('⚡ Initialisation Scalper Pro TradingView...');
        
        this.setupTradingView();
        this.setupWebSocket();
        this.setupEventListeners();
        this.updateServerTime();
        
        setInterval(() => this.updateServerTime(), 1000);
        
        // Charger indicateurs toutes les 5s (fallback si WS déconnecté)
        setInterval(() => this.loadIndicators(), 5000);
    }

    // ========== TRADINGVIEW WIDGET ==========
    setupTradingView() {
        console.log('📈 Chargement TradingView Widget...');
        
        this.widget = new TradingView.widget({
            container_id: 'tradingview_chart',
            autosize: true,
            symbol: `NASDAQ:${this.currentTicker}`,
            interval: this.currentTimeframe,
            timezone: 'America/New_York',
            theme: 'dark',
            style: '1', // Bougie
            locale: 'fr',
            toolbar_bg: '#151b3d',
            enable_publishing: false,
            hide_side_toolbar: false,
            allow_symbol_change: true,
            save_image: false,
            
            // Indicateurs pré-chargés
            studies: [
                { id: 'RSI@tv-basicstudies', inputs: { length: 14 } },
                { id: 'MACD@tv-basicstudies', inputs: { fastLength: 12, slowLength: 26, signalLength: 9 } },
                { id: 'BB@tv-basicstudies', inputs: { length: 20, mult: 2 } },
                { id: 'MASimple@tv-basicstudies', inputs: { length: 20 } },
                { id: 'MASimple@tv-basicstudies', inputs: { length: 50 } },
                { id: 'Volume@tv-basicstudies' }
            ],
            
            // Design
            overrides: {
                'paneProperties.background': '#0a0e27',
                'paneProperties.backgroundType': 'solid',
                'mainSeriesProperties.candleStyle.upColor': '#00ff88',
                'mainSeriesProperties.candleStyle.downColor': '#ff3366',
                'mainSeriesProperties.candleStyle.borderUpColor': '#00ff88',
                'mainSeriesProperties.candleStyle.borderDownColor': '#ff3366',
                'mainSeriesProperties.candleStyle.wickUpColor': '#00ff88',
                'mainSeriesProperties.candleStyle.wickDownColor': '#ff3366'
            },
            
            // Features activées
            enabled_features: [
                'study_templates',
                'use_localstorage_for_settings',
                'save_chart_properties_to_local_storage'
            ],
            
            // Features désactivées
            disabled_features: [
                'header_symbol_search',
                'header_compare',
                'display_market_status'
            ],
            
            // Custom CSS
            custom_css_url: '/static/css/tradingview_custom.css',
            
            // Loading screen
            loading_screen: { 
                backgroundColor: '#0a0e27',
                foregroundColor: '#00d4ff'
            }
        });
        
        console.log('✅ TradingView Widget chargé');
    }

    changeTicker(ticker) {
        this.currentTicker = ticker.toUpperCase();
        document.getElementById('tickerSymbol').textContent = this.currentTicker;
        
        if (this.widget) {
            this.widget.setSymbol(`NASDAQ:${this.currentTicker}`, this.currentTimeframe, () => {
                console.log(`✅ Ticker changé: ${this.currentTicker}`);
            });
        }
        
        // Mettre à jour WebSocket subscription
        if (this.socket && this.socket.connected) {
            this.socket.emit('subscribe', { ticker: this.currentTicker });
        }
        
        this.loadIndicators();
    }

    changeTimeframe(tf) {
        this.currentTimeframe = tf;
        
        if (this.widget) {
            this.widget.setSymbol(`NASDAQ:${this.currentTicker}`, tf, () => {
                console.log(`✅ Timeframe changé: ${tf}`);
            });
        }
    }

    // ========== WEBSOCKET ==========
    setupWebSocket() {
        console.log('🔌 Connexion WebSocket...');
        
        // Connexion au serveur Flask-SocketIO
        this.socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: this.maxReconnectAttempts
        });
        
        // Événements connexion
        this.socket.on('connect', () => {
            console.log('✅ WebSocket connecté');
            this.updateWSStatus(true);
            this.reconnectAttempts = 0;
            
            // S'abonner au ticker actuel
            this.socket.emit('subscribe', { ticker: this.currentTicker });
        });
        
        this.socket.on('disconnect', (reason) => {
            console.log('⚠️ WebSocket déconnecté:', reason);
            this.updateWSStatus(false);
        });
        
        this.socket.on('reconnect_attempt', (attemptNumber) => {
            console.log(`🔄 Tentative de reconnexion ${attemptNumber}/${this.maxReconnectAttempts}`);
            this.reconnectAttempts = attemptNumber;
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('❌ Erreur WebSocket:', error);
        });
        
        // Événements données
        this.socket.on('price_update', (data) => {
            this.handlePriceUpdate(data);
        });
        
        this.socket.on('indicator_update', (data) => {
            this.handleIndicatorUpdate(data);
        });
    }

    updateWSStatus(connected) {
        const dot = document.getElementById('wsDot');
        const status = document.getElementById('wsStatus');
        
        if (connected) {
            dot.classList.remove('disconnected');
            status.textContent = 'Connected';
        } else {
            dot.classList.add('disconnected');
            status.textContent = 'Disconnected';
        }
    }

    handlePriceUpdate(data) {
        if (data.ticker !== this.currentTicker) return;
        
        // Mettre à jour le prix ticker
        const price = data.price;
        const change = data.change;
        const changePct = data.change_pct;
        
        document.getElementById('tickerPrice').textContent = `$${price.toFixed(2)}`;
        
        const changeEl = document.getElementById('tickerChange');
        changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)`;
        changeEl.className = change >= 0 ? 'ticker-change positive' : 'ticker-change negative';
        
        // Mettre à jour les stats
        if (data.volume) {
            document.getElementById('statVolume').textContent = this.formatVolume(data.volume);
        }
        if (data.high) {
            document.getElementById('statHigh').textContent = `$${data.high.toFixed(2)}`;
        }
        if (data.low) {
            document.getElementById('statLow').textContent = `$${data.low.toFixed(2)}`;
        }
        if (data.open) {
            document.getElementById('statOpen').textContent = `$${data.open.toFixed(2)}`;
        }
    }

    handleIndicatorUpdate(data) {
        if (data.ticker !== this.currentTicker) return;
        
        this.indicators = data.indicators;
        this.updateIndicatorsDisplay();
    }

    // ========== INDICATEURS ==========
    async loadIndicators() {
        try {
            const response = await fetch(`/api/pro-analysis/${this.currentTicker}`);
            const data = await response.json();
            
            if (data && !data.error) {
                this.indicators = {
                    rsi: data.momentum?.rsi_value,
                    macd: data.momentum?.macd_value,
                    macd_signal: data.momentum?.macd_signal,
                    stoch: data.momentum?.stoch_k,
                    adx: data.trend?.adx_value,
                    atr: data.volatility?.atr_value
                };
                
                this.updateIndicatorsDisplay();
            }
        } catch (error) {
            console.error('Erreur chargement indicateurs:', error);
        }
    }

    updateIndicatorsDisplay() {
        const ind = this.indicators;
        
        // RSI
        if (ind.rsi !== undefined && ind.rsi !== null) {
            document.getElementById('indRSI').textContent = ind.rsi.toFixed(1);
            this.updateIndicatorSignal('sigRSI', this.getRSISignal(ind.rsi));
        }
        
        // MACD
        if (ind.macd !== undefined && ind.macd !== null) {
            document.getElementById('indMACD').textContent = ind.macd.toFixed(2);
            if (ind.macd_signal !== undefined) {
                const signal = ind.macd > ind.macd_signal ? 'buy' : 'sell';
                this.updateIndicatorSignal('sigMACD', signal);
            }
        }
        
        // Stochastique
        if (ind.stoch !== undefined && ind.stoch !== null) {
            document.getElementById('indSTOCH').textContent = ind.stoch.toFixed(1);
            this.updateIndicatorSignal('sigSTOCH', this.getStochSignal(ind.stoch));
        }
        
        // ADX
        if (ind.adx !== undefined && ind.adx !== null) {
            document.getElementById('indADX').textContent = ind.adx.toFixed(1);
            const adxText = ind.adx > 25 ? 'FORT' : 'FAIBLE';
            document.getElementById('sigADX').textContent = adxText;
        }
        
        // ATR
        if (ind.atr !== undefined && ind.atr !== null) {
            document.getElementById('indATR').textContent = ind.atr.toFixed(2);
        }
    }

    getRSISignal(rsi) {
        if (rsi < 30) return 'buy';
        if (rsi > 70) return 'sell';
        return 'neutral';
    }

    getStochSignal(stoch) {
        if (stoch < 20) return 'buy';
        if (stoch > 80) return 'sell';
        return 'neutral';
    }

    updateIndicatorSignal(elementId, signal) {
        const el = document.getElementById(elementId);
        el.className = `indicator-signal signal-${signal}`;
        
        const text = {
            'buy': 'ACHAT',
            'sell': 'VENTE',
            'neutral': 'NEUTRE'
        };
        
        el.textContent = text[signal] || signal.toUpperCase();
    }

    // ========== EVENT LISTENERS ==========
    setupEventListeners() {
        // Changement de ticker
        const tickerInput = document.getElementById('tickerInput');
        tickerInput.addEventListener('change', (e) => {
            this.changeTicker(e.target.value);
        });
        
        tickerInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.changeTicker(e.target.value);
            }
        });
        
        // Changement de timeframe
        document.querySelectorAll('.timeframe-pill').forEach(pill => {
            pill.addEventListener('click', (e) => {
                document.querySelectorAll('.timeframe-pill').forEach(p => p.classList.remove('active'));
                e.target.classList.add('active');
                this.changeTimeframe(e.target.dataset.tf);
            });
        });
        
        // Quick trading buttons
        document.getElementById('btnQuickBuy').addEventListener('click', () => this.quickTrade('BUY'));
        document.getElementById('btnQuickSell').addEventListener('click', () => this.quickTrade('SELL'));
        
        // Order panel tabs
        document.querySelectorAll('.order-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                document.querySelectorAll('.order-tab').forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                
                const orderType = e.target.dataset.type;
                const limitPriceGroup = document.getElementById('limitPriceGroup');
                
                if (orderType === 'limit' || orderType === 'stop') {
                    limitPriceGroup.style.display = 'block';
                } else {
                    limitPriceGroup.style.display = 'none';
                }
            });
        });
        
        // Order submit buttons
        document.getElementById('btnSubmitBuy').addEventListener('click', () => this.submitOrder('BUY'));
        document.getElementById('btnSubmitSell').addEventListener('click', () => this.submitOrder('SELL'));
    }

    // ========== TRADING ==========
    quickTrade(side) {
        const confirmation = confirm(`⚠️ Confirmer ${side} ${this.currentTicker} ?`);
        if (!confirmation) return;
        
        // TODO: Intégration Alpaca API
        console.log(`⚡ Quick ${side} pour ${this.currentTicker}`);
        
        alert(`⚡ Order ${side} placé pour ${this.currentTicker}\n(Intégration Alpaca en cours...)`);
    }

    submitOrder(side) {
        const qty = document.getElementById('orderQty').value;
        const orderType = document.querySelector('.order-tab.active').dataset.type;
        
        let orderData = {
            ticker: this.currentTicker,
            side: side,
            qty: parseInt(qty),
            type: orderType
        };
        
        if (orderType === 'limit' || orderType === 'stop') {
            const limitPrice = document.getElementById('orderLimitPrice').value;
            if (!limitPrice) {
                alert('⚠️ Veuillez entrer un prix limite');
                return;
            }
            orderData.limit_price = parseFloat(limitPrice);
        }
        
        console.log('📤 Ordre soumis:', orderData);
        
        // TODO: Envoyer via WebSocket ou API
        alert(`⚡ Ordre ${side} soumis\nTicker: ${orderData.ticker}\nQty: ${orderData.qty}\nType: ${orderData.type}`);
    }

    // ========== UTILS ==========
    updateServerTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('fr-FR');
        document.getElementById('serverTime').textContent = timeStr;
    }

    formatVolume(vol) {
        if (vol >= 1e9) return (vol / 1e9).toFixed(2) + 'B';
        if (vol >= 1e6) return (vol / 1e6).toFixed(2) + 'M';
        if (vol >= 1e3) return (vol / 1e3).toFixed(2) + 'K';
        return vol.toString();
    }
}

// ========== INIT ==========
document.addEventListener('DOMContentLoaded', () => {
    window.scalperTV = new ScalperTradingView();
});
