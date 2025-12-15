"""
🔥 LIVE TRADING ROUTES - API Flask pour le dashboard temps réel

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
    LIVE_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ LiveAnalyzer non disponible: {e}")
    LIVE_ANALYZER_AVAILABLE = False


live_bp = Blueprint('live', __name__, url_prefix='/api/live')

# Global state
live_analyzer = None
monitoring_thread = None
websocket_thread = None
signal_queue = Queue(maxsize=100)  # Queue pour SSE


@live_bp.route('/start', methods=['POST'])
def start_monitoring():
    """
    Démarre le monitoring temps réel
    
    Body:
        {
            "tickers": ["NVDA", "AAPL", "TSLA"],
            "timeframe": 1  # minutes
        }
    """
    if not LIVE_ANALYZER_AVAILABLE:
        return jsonify({
            'error': 'LiveAnalyzer non disponible',
            'message': 'Module streaming.live_analyzer non installé'
        }), 503
    
    global live_analyzer, monitoring_thread, websocket_thread
    
    if live_analyzer is not None:
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
        
        # === UTILISER LE SINGLETON WEBSOCKET ===
        ws_manager = WebSocketManager.get_instance()
        
        # Créer l'analyzer SANS créer de connexion WebSocket interne
        live_analyzer = LiveAnalyzer(tickers, timeframe_minutes=timeframe, use_websocket_manager=True)
        
        # Ajouter callback pour envoyer les signaux vers SSE
        def on_signal(signal):
            """Callback appelé à chaque signal détecté"""
            try:
                signal_queue.put({
                    'type': 'signal',
                    'data': signal
                })
            except Exception as e:
                logger.error(f"❌ Erreur ajout signal à queue: {e}")
        
        live_analyzer.add_signal_callback(on_signal)
        
        # === DÉMARRER LE WEBSOCKET (UNE SEULE FOIS) ===
        if not ws_manager.is_running:
            # Utiliser la helper function qui gère correctement l'event loop
            websocket_thread = start_websocket_in_thread(ws_manager)
            logger.info("🚀 WebSocket Alpaca démarré (singleton)")
        else:
            logger.info("🔗 WebSocket déjà actif (réutilisation)")
        
        # Démarrer l'analyzer dans un thread séparé
        def run_analyzer():
            try:
                # Créer un nouvel event loop pour ce thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(live_analyzer.start())
            except Exception as e:
                logger.error(f"❌ Erreur run analyzer: {e}", exc_info=True)
            finally:
                loop.close()
        
        monitoring_thread = threading.Thread(target=run_analyzer, daemon=True, name="LiveAnalyzer")
        monitoring_thread.start()
        
        logger.info(f"✅ Monitoring démarré pour {len(tickers)} tickers (timeframe: {timeframe}min)")
        
        return jsonify({
            'status': 'started',
            'tickers': tickers,
            'timeframe': timeframe,
            'websocket_shared': True,
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
    global live_analyzer, monitoring_thread
    
    if live_analyzer is None:
        return jsonify({
            'error': 'Aucun monitoring actif',
            'message': 'Le monitoring n\'est pas démarré'
        }), 400
    
    try:
        # Arrêter l'analyzer
        live_analyzer.stop()
        
        # Nettoyer
        live_analyzer = None
        monitoring_thread = None
        
        # Vider la queue
        while not signal_queue.empty():
            try:
                signal_queue.get_nowait()
            except:
                break
        
        # NOTE: On ne stop PAS le WebSocket, il reste actif pour d'autres instances
        
        logger.info("✅ Monitoring arrêté")
        
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
    if live_analyzer is None:
        return jsonify({
            'monitoring': False,
            'tickers': [],
            'stats': {}
        })
    
    try:
        state = live_analyzer.get_current_state()
        
        # Stats du WebSocket Manager
        try:
            ws_manager = WebSocketManager.get_instance()
            ws_stats = ws_manager.get_stats()
        except:
            ws_stats = {'error': 'WebSocket non initialisé'}
        
        return jsonify({
            'monitoring': True,
            'tickers': live_analyzer.tickers,
            'timeframe': live_analyzer.timeframe_minutes,
            'start_time': live_analyzer.start_time.isoformat() if live_analyzer.start_time else None,
            'total_bars': live_analyzer.total_bars_received,
            'total_signals': live_analyzer.total_signals_generated,
            'state': state,
            'websocket': ws_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur get state: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_bp.route('/stream')
def stream_signals():
    """
    Server-Sent Events (SSE) pour recevoir les signaux en temps réel
    
    Utilisation frontend:
        const eventSource = new EventSource('/api/live/stream');
        eventSource.addEventListener('signal', (event) => {
            const signal = JSON.parse(event.data);
            console.log(signal);
        });
    """
    def generate():
        """Générateur SSE"""
        # Envoyer heartbeat toutes les 30 secondes
        import time
        last_heartbeat = time.time()
        
        while True:
            try:
                # Vérifier si des signaux sont disponibles
                if not signal_queue.empty():
                    item = signal_queue.get(timeout=1)
                    
                    event_type = item.get('type', 'signal')
                    data = item.get('data', {})
                    
                    # Formater en SSE
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(data, default=str)}\n\n"
                    
                else:
                    # Heartbeat pour garder la connexion ouverte
                    current_time = time.time()
                    if current_time - last_heartbeat > 30:
                        yield f"event: heartbeat\n"
                        yield f"data: {{\"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                        last_heartbeat = current_time
                    
                    # Attendre un peu avant de vérifier à nouveau
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"❌ Erreur SSE generator: {e}")
                break
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Pour Nginx
        }
    )


@live_bp.route('/health')
def health():
    """
    Endpoint de santé
    """
    try:
        ws_manager = WebSocketManager.get_instance()
        ws_stats = ws_manager.get_stats()
    except:
        ws_stats = {'error': 'WebSocketManager non initialisé'}
    
    return jsonify({
        'status': 'healthy',
        'analyzer_available': LIVE_ANALYZER_AVAILABLE,
        'monitoring': live_analyzer is not None,
        'queue_size': signal_queue.qsize(),
        'websocket': ws_stats,
        'timestamp': datetime.now().isoformat()
    })


@live_bp.route('/websocket/reset', methods=['POST'])
def reset_websocket():
    """
    REDÉMARRAGE FORCÉ du WebSocket (debug uniquement)
    ⚠️  Utiliser seulement en cas de problème
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
