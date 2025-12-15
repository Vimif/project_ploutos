#!/usr/bin/env python3
"""
⚡ WEBSOCKET SERVER - PRIX TEMPS RÉEL
Flask-SocketIO + Alpaca Data Stream API
"""

import os
import sys
import logging
import time
from threading import Thread, Event
from datetime import datetime

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS

import yfinance as yf

try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.data.historical import StockHistoricalDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ alpaca-py non installé, mode DEMO uniquement")
    ALPACA_AVAILABLE = False

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'ploutos-secret-key-2024')
CORS(app)

# Socket.IO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=True,
    engineio_logger=False
)

# Alpaca credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.warning("⚠️ Alpaca credentials manquantes, mode démo activé")
    DEMO_MODE = True
else:
    DEMO_MODE = False

# Gestion des abonnements
active_subscriptions = {}
alpaca_stream = None
stream_thread = None
stream_stop_event = Event()


# ========== WEBSOCKET EVENTS ==========

@socketio.on('connect')
def handle_connect():
    """Client connecté"""
    logger.info(f"✅ Client connecté: {request.sid}")
    emit('connected', {'status': 'success', 'demo_mode': DEMO_MODE})


@socketio.on('disconnect')
def handle_disconnect():
    """Client déconnecté"""
    logger.info(f"❌ Client déconnecté: {request.sid}")
    
    # Retirer des abonnements
    for ticker in list(active_subscriptions.keys()):
        if request.sid in active_subscriptions[ticker]:
            active_subscriptions[ticker].remove(request.sid)
            
            # Si plus personne abonné, retirer le ticker
            if len(active_subscriptions[ticker]) == 0:
                del active_subscriptions[ticker]
                logger.info(f"🗑️ Ticker {ticker} retiré (plus d'abonnés)")


@socketio.on('subscribe')
def handle_subscribe(data):
    """Abonnement à un ticker"""
    ticker = data.get('ticker', 'AAPL').upper()
    
    # Ajouter le client à la room du ticker
    join_room(ticker)
    
    # Ajouter aux abonnements actifs
    if ticker not in active_subscriptions:
        active_subscriptions[ticker] = []
    
    if request.sid not in active_subscriptions[ticker]:
        active_subscriptions[ticker].append(request.sid)
    
    logger.info(f"📶 {request.sid} abonné à {ticker} ({len(active_subscriptions[ticker])} clients)")
    
    # Envoyer prix initial
    send_initial_price(ticker)
    
    emit('subscribed', {'ticker': ticker, 'status': 'success'})


@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Désabonnement d'un ticker"""
    ticker = data.get('ticker', '').upper()
    
    leave_room(ticker)
    
    if ticker in active_subscriptions and request.sid in active_subscriptions[ticker]:
        active_subscriptions[ticker].remove(request.sid)
        logger.info(f"🚫 {request.sid} désabonné de {ticker}")
        
        if len(active_subscriptions[ticker]) == 0:
            del active_subscriptions[ticker]
    
    emit('unsubscribed', {'ticker': ticker, 'status': 'success'})


# ========== ALPACA DATA STREAM ==========

def send_initial_price(ticker):
    """Envoyer prix initial depuis yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose', current_price)
        
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close > 0 else 0
        
        data = {
            'ticker': ticker,
            'price': current_price,
            'change': change,
            'change_pct': change_pct,
            'volume': info.get('volume', 0),
            'high': info.get('dayHigh', current_price),
            'low': info.get('dayLow', current_price),
            'open': info.get('open', current_price),
            'timestamp': datetime.now().isoformat()
        }
        
        socketio.emit('price_update', data, room=ticker)
        logger.info(f"📈 Prix initial envoyé pour {ticker}: ${current_price:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Erreur prix initial {ticker}: {e}")


def start_alpaca_stream():
    """Démarrer le stream Alpaca (thread séparé)"""
    global alpaca_stream
    
    if DEMO_MODE:
        logger.info("🛠️ Mode DEMO - Stream Alpaca désactivé")
        start_demo_stream()
        return
    
    try:
        logger.info("🚀 Démarrage Alpaca Data Stream...")
        
        alpaca_stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )
        
        # Handler pour les trades
        async def trade_handler(trade):
            ticker = trade.symbol
            
            if ticker in active_subscriptions:
                data = {
                    'ticker': ticker,
                    'price': trade.price,
                    'volume': trade.size,
                    'timestamp': trade.timestamp.isoformat()
                }
                
                socketio.emit('price_update', data, room=ticker)
        
        # Handler pour les quotes
        async def quote_handler(quote):
            ticker = quote.symbol
            
            if ticker in active_subscriptions:
                mid_price = (quote.bid_price + quote.ask_price) / 2
                
                data = {
                    'ticker': ticker,
                    'price': mid_price,
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'spread': quote.ask_price - quote.bid_price,
                    'timestamp': quote.timestamp.isoformat()
                }
                
                socketio.emit('price_update', data, room=ticker)
        
        # Enregistrer handlers
        alpaca_stream.subscribe_trades(trade_handler, *active_subscriptions.keys())
        alpaca_stream.subscribe_quotes(quote_handler, *active_subscriptions.keys())
        
        # Lancer le stream
        logger.info("✅ Alpaca Stream démarré")
        alpaca_stream.run()
        
    except Exception as e:
        logger.error(f"❌ Erreur Alpaca Stream: {e}")
        logger.info("🔄 Fallback vers mode DEMO")
        start_demo_stream()


def start_demo_stream():
    """Stream de démo utilisant yfinance (polling)"""
    logger.info("🎯 Démarrage stream DEMO (yfinance polling)")
    
    while not stream_stop_event.is_set():
        try:
            for ticker in list(active_subscriptions.keys()):
                if len(active_subscriptions[ticker]) == 0:
                    continue
                
                # Récupérer prix yfinance
                stock = yf.Ticker(ticker)
                info = stock.info
                
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                prev_close = info.get('previousClose', current_price)
                
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                
                data = {
                    'ticker': ticker,
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct,
                    'volume': info.get('volume', 0),
                    'high': info.get('dayHigh', current_price),
                    'low': info.get('dayLow', current_price),
                    'open': info.get('open', current_price),
                    'timestamp': datetime.now().isoformat()
                }
                
                socketio.emit('price_update', data, room=ticker)
            
            # Attendre 2 secondes avant le prochain update
            stream_stop_event.wait(2)
            
        except Exception as e:
            logger.error(f"❌ Erreur DEMO stream: {e}")
            stream_stop_event.wait(5)


# ========== ROUTES FLASK ==========

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'websocket': 'active',
        'demo_mode': DEMO_MODE,
        'active_tickers': list(active_subscriptions.keys()),
        'total_clients': sum(len(clients) for clients in active_subscriptions.values())
    }


@app.route('/subscribers')
def subscribers():
    return {
        'subscriptions': {
            ticker: len(clients) 
            for ticker, clients in active_subscriptions.items()
        }
    }


# ========== MAIN ==========

if __name__ == '__main__':
    # Démarrer stream dans un thread séparé
    stream_thread = Thread(target=start_alpaca_stream, daemon=True)
    stream_thread.start()
    
    # Démarrer serveur WebSocket
    logger.info("⚡ Démarrage WebSocket Server sur port 5001...")
    
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5001,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️ Arrêt du serveur...")
        stream_stop_event.set()
        sys.exit(0)
