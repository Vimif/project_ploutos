"""
🔥 LIVE TRADING ROUTES - API Flask pour le dashboard temps réel

Supporte 2 sources de données :
- Alpaca WebSocket : Actions US (NVDA, AAPL, TSLA, etc.)
- Yahoo Finance Polling : Actions européennes (DSY.PA, MC.PA, AI.PA, etc.)

Détection automatique : Si ticker contient '.PA', '.L', '.DE' → Yahoo, sinon Alpaca

Routes :
- POST /api/live/start : Démarre le monitoring
- POST /api/live/stop : Arrête le monitoring
- GET /api/live/state : État actuel
- GET /api/live/stream : Server-Sent Events (SSE) pour signaux temps réel
"""

import asyncio
import threading
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, Response
from queue import Queue
import logging

logger = logging.getLogger(__name__)

try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from streaming.live_analyzer import LiveAnalyzer
    from streaming.websocket_manager import WebSocketManager, start_websocket_in_thread
    from streaming.yahoo_stream import YahooStreamManager
    LIVE_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ LiveAnalyzer non disponible: {e}")
    LIVE_ANALYZER_AVAILABLE = False


live_bp = Blueprint('live', __name__, url_prefix='/api/live')

# Global state
live_analyzer = None
monitoring_thread = None
websocket_thread = None
yahoo_manager = None
signal_queue = Queue(maxsize=100)  # Queue pour SSE


def is_european_ticker(ticker: str) -> bool:
    """
    Détecte si un ticker est européen.
    
    Exemples:
        DSY.PA  → True  (Euronext Paris)
        MC.PA   → True  (Euronext Paris)
        VOD.L   → True  (London Stock Exchange)
        SAP.DE  → True  (Frankfurt)
        NVDA    → False (US)
        AAPL    → False (US)
    """
    european_suffixes = ['.PA', '.L', '.DE', '.AS', '.BR', '.MI', '.MC']
    return any(ticker.upper().endswith(suffix) for suffix in european_suffixes)


@live_bp.route('/start', methods=['POST'])
def start_monitoring():
    """
    Démarre le monitoring temps réel
    
    Body:
        {
            "tickers": ["NVDA", "DSY.PA", "AAPL"],
            "timeframe": 1  # minutes (ignoré pour Yahoo)
        }
    """
    if not LIVE_ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'LiveAnalyzer non disponible',
            'message': 'Module streaming.live_analyzer non installé'
        }), 503
    
    global live_analyzer, monitoring_thread, websocket_thread, yahoo_manager
    
    if live_analyzer is not None or yahoo_manager is not None:
        return jsonify({
            'error': 'Monitoring déjà actif',
            'message': 'Arrêtez d\'abord le monitoring en cours'
        }), 400
    
    try:
        data = request.json or {}
        tickers = data.get('tickers', ['NVDA', 'AAPL', 'TSLA'])
        timeframe = int(data.get('timeframe', 1))
        
        # Valider les paramètres
        if not tickers or len(tickers) == 0:
            return jsonify({'error': 'Au moins un ticker requis'}), 400
        
        if timeframe not in [1, 5, 15, 30, 60]:
            return jsonify({'error': 'Timeframe invalide (1, 5, 15, 30, 60 minutes)'}), 400
        
        # === SÉPARER TICKERS US / EUROPÉENS ===
        us_tickers = [t for t in tickers if not is_european_ticker(t)]
        eu_tickers = [t for t in tickers if is_european_ticker(t)]
        
        logger.info(f"🇺🇸 Tickers US: {us_tickers}")
        logger.info(f"🇪🇺 Tickers EU: {eu_tickers}")
        
        # Callback commun pour envoyer les signaux vers SSE
        def on_signal(signal):
            """Callback appelé à chaque signal détecté"""
            try:
                signal_queue.put({
                    'type': 'signal',
                    'data': signal
                })
            except Exception as e:
                logger.error(f"❌ Erreur ajout signal à queue: {e}")
        
        # Callback pour barres (stats uniquement)
        def on_bar(bar):
            """Callback pour chaque barre reçue"""
            try:
                signal_queue.put({
                    'type': 'bar',
                    'data': {
                        'symbol': bar.symbol,
                        'close': bar.close,
                        'timestamp': str(bar.timestamp)
                    }
                })
            except Exception as e:
                logger.error(f"❌ Erreur ajout barre à queue: {e}")
        
        # === DÉMARRER ALPACA (SI TICKERS US) ===
        if us_tickers:
            ws_manager = WebSocketManager.get_instance()
            
            # Créer l'analyzer
            live_analyzer = LiveAnalyzer(us_tickers, timeframe_minutes=timeframe, use_websocket_manager=True)
            live_analyzer.add_signal_callback(on_signal)
            
            # Démarrer le WebSocket (une seule fois)
            if not ws_manager.is_running:
                websocket_thread = start_websocket_in_thread(ws_manager)
                logger.info("🚀 WebSocket Alpaca démarré (singleton)")
            else:
                logger.info("🔗 WebSocket déjà actif (réutilisation)")
            
            # Démarrer l'analyzer dans un thread
            def run_analyzer():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(live_analyzer.start())
                except Exception as e:
                    logger.error(f"❌ Erreur run analyzer: {e}", exc_info=True)
                finally:
                    loop.close()
            
            monitoring_thread = threading.Thread(target=run_analyzer, daemon=True, name="LiveAnalyzer")
            monitoring_thread.start()
            logger.info(f"✅ Alpaca monitoring démarré pour {len(us_tickers)} ticker(s) US")
        
        # === DÉMARRER YAHOO (SI TICKERS EUROPÉENS) ===
        if eu_tickers:
            # Convertir timeframe en intervalle de polling (min 10s)
            poll_interval = max(timeframe * 60, 10)  # timeframe en minutes → secondes
            
            yahoo_manager = YahooStreamManager(default_interval=poll_interval)
            
            # S'abonner à chaque ticker
            for ticker in eu_tickers:
                yahoo_manager.subscribe(ticker, on_bar)
            
            # Démarrer le polling
            yahoo_manager.start()
            logger.info(f"✅ Yahoo polling démarré pour {len(eu_tickers)} ticker(s) EU (interval: {poll_interval}s)")
        
        return jsonify({
            'status': 'started',
            'tickers_us': us_tickers,
            'tickers_eu': eu_tickers,
            'timeframe': timeframe,
            'sources': {
                'alpaca': len(us_tickers) > 0,
                'yahoo': len(eu_tickers) > 0
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur start monitoring: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_bp.route('/stop', methods=['POST'])
def stop_monitoring():
    """
    Arrête le monitoring en cours
    """
    global live_analyzer, monitoring_thread, yahoo_manager
    
    if live_analyzer is None and yahoo_manager is None:
        return jsonify({
            'error': 'Aucun monitoring actif',
            'message': 'Le monitoring n\'est pas démarré'
        }), 400
    
    try:
        # Arrêter Alpaca
        if live_analyzer is not None:
            live_analyzer.stop()
            live_analyzer = None
            monitoring_thread = None
            logger.info("✅ Alpaca monitoring arrêté")
        
        # Arrêter Yahoo
        if yahoo_manager is not None:
            yahoo_manager.stop()
            yahoo_manager = None
            logger.info("✅ Yahoo polling arrêté")
        
        # Vider la queue
        while not signal_queue.empty():
            try:
                signal_queue.get_nowait()
            except:
                break
        
        return jsonify({
            'status': 'stopped',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur stop monitoring: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_bp.route('/state')
def get_state():
    """
    Récupère l'état actuel du monitoring
    """
    if live_analyzer is None and yahoo_manager is None:
        return jsonify({
            'monitoring': False,
            'tickers': [],
            'stats': {}
        })
    
    try:
        state = {}
        
        # Stats Alpaca
        if live_analyzer is not None:
            state['alpaca'] = live_analyzer.get_current_state()
            state['tickers_us'] = live_analyzer.tickers
            state['timeframe'] = live_analyzer.timeframe_minutes
        
        # Stats Yahoo
        if yahoo_manager is not None:
            state['yahoo'] = yahoo_manager.get_stats()
            state['tickers_eu'] = list(yahoo_manager.subscriptions.keys())
        
        # Stats WebSocket
        try:
            ws_manager = WebSocketManager.get_instance()
            state['websocket'] = ws_manager.get_stats()
        except:
            state['websocket'] = {'error': 'WebSocket non initialisé'}
        
        return jsonify({
            'monitoring': True,
            'state': state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur get state: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_bp.route('/stream')
def stream_signals():
    """
    Server-Sent Events (SSE) pour recevoir les signaux en temps réel
    """
    def generate():
        import time
        last_heartbeat = time.time()
        
        while True:
            try:
                if not signal_queue.empty():
                    item = signal_queue.get(timeout=1)
                    event_type = item.get('type', 'signal')
                    data = item.get('data', {})
                    
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(data, default=str)}\n\n"
                else:
                    current_time = time.time()
                    if current_time - last_heartbeat > 30:
                        yield f"event: heartbeat\n"
                        yield f"data: {{\"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                        last_heartbeat = current_time
                    
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"❌ Erreur SSE generator: {e}")
                break
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@live_bp.route('/health')
def health():
    """
    Endpoint de santé
    """
    try:
        ws_stats = WebSocketManager.get_instance().get_stats()
    except:
        ws_stats = {}
    
    try:
        yahoo_stats = yahoo_manager.get_stats() if yahoo_manager else {}
    except:
        yahoo_stats = {}
    
    return jsonify({
        'status': 'healthy',
        'analyzer_available': LIVE_ANALYZER_AVAILABLE,
        'monitoring_alpaca': live_analyzer is not None,
        'monitoring_yahoo': yahoo_manager is not None,
        'queue_size': signal_queue.qsize(),
        'websocket': ws_stats,
        'yahoo': yahoo_stats,
        'timestamp': datetime.now().isoformat()
    })


@live_bp.route('/websocket/reset', methods=['POST'])
def reset_websocket():
    """
    REDÉMARRAGE FORCÉ du WebSocket (debug uniquement)
    """
    global websocket_thread
    
    try:
        WebSocketManager.reset_instance()
        websocket_thread = None
        logger.warning("♻️ WebSocket redémarré (reset forcé)")
        return jsonify({
            'status': 'reset',
            'message': 'WebSocket redémarré avec succès',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
