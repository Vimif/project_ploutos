"""
📊 LIVE TRADING WATCHLISTS - Gestion des watchlists pour live trading

Routes :
- GET /api/live/watchlists : Liste toutes les watchlists
- POST /api/live/watchlists : Crée une nouvelle watchlist
- DELETE /api/live/watchlists/<id> : Supprime une watchlist
"""

import json
from pathlib import Path
from flask import Blueprint, jsonify, request
import logging

# Utiliser les settings existants
try:
    from config.settings import DATA_DIR
except ImportError:
    DATA_DIR = Path('data')

logger = logging.getLogger(__name__)

live_watchlists_bp = Blueprint('live_watchlists', __name__, url_prefix='/api/live')

# Fichier de stockage (utilise DATA_DIR du .env)
WATCHLISTS_FILE = DATA_DIR / 'live_watchlists.json'
WATCHLISTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Watchlists prédéfinies
DEFAULT_WATCHLISTS = [
    {
        'id': 'us_tech_giants',
        'name': '🇺🇸 US Tech Giants',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        'description': 'Les 7 géants de la tech US',
        'editable': False
    },
    {
        'id': 'us_mega_cap',
        'name': '📈 US Mega Cap',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK.B', 'META', 'TSLA', 'V', 'JNJ'],
        'description': 'Top 10 capitalisations US',
        'editable': False
    },
    {
        'id': 'us_ai_revolution',
        'name': '🤖 AI Revolution',
        'tickers': ['NVDA', 'MSFT', 'GOOGL', 'META', 'AMD', 'PLTR', 'AI', 'SNOW'],
        'description': 'Actions IA et cloud',
        'editable': False
    },
    {
        'id': 'us_finance',
        'name': '🏦 US Finance',
        'tickers': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW'],
        'description': 'Banques et services financiers US',
        'editable': False
    },
    {
        'id': 'us_energy',
        'name': '⚡ US Energy',
        'tickers': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'HAL'],
        'description': 'Énergie et pétrole US',
        'editable': False
    },
    {
        'id': 'us_etf_major',
        'name': '📉 US ETF Major',
        'tickers': ['SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'DIA', 'IVV', 'VEA'],
        'description': 'ETF principaux US',
        'editable': False
    },
    {
        'id': 'us_consumer',
        'name': '🛍️ US Consumer',
        'tickers': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
        'description': 'Consommation et retail US',
        'editable': False
    },
    {
        'id': 'us_healthcare',
        'name': '⚕️ US Healthcare',
        'tickers': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR'],
        'description': 'Santé et pharma US',
        'editable': False
    },
    {
        'id': 'quick_5',
        'name': '⚡ Quick 5',
        'tickers': ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL'],
        'description': 'Top 5 pour tests rapides',
        'editable': False
    }
]


def load_watchlists():
    """Charge les watchlists depuis le fichier"""
    if not WATCHLISTS_FILE.exists():
        return DEFAULT_WATCHLISTS.copy()
    
    try:
        with open(WATCHLISTS_FILE, 'r', encoding='utf-8') as f:
            custom = json.load(f)
        
        # Fusionner prédéfinies + custom
        return DEFAULT_WATCHLISTS + custom
    except Exception as e:
        logger.error(f"❌ Erreur chargement watchlists: {e}")
        return DEFAULT_WATCHLISTS.copy()


def save_custom_watchlists(watchlists):
    """Sauvegarde les watchlists personnalisées"""
    # Ne sauvegarder que les watchlists éditables
    custom = [wl for wl in watchlists if wl.get('editable', True)]
    
    try:
        with open(WATCHLISTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(custom, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde watchlists: {e}")
        return False


@live_watchlists_bp.route('/watchlists')
def get_watchlists():
    """
    Récupère toutes les watchlists (prédéfinies + custom)
    """
    try:
        watchlists = load_watchlists()
        
        return jsonify({
            'watchlists': watchlists,
            'count': len(watchlists)
        })
    except Exception as e:
        logger.error(f"❌ Erreur get watchlists: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_watchlists_bp.route('/watchlists', methods=['POST'])
def create_watchlist():
    """
    Crée une nouvelle watchlist personnalisée
    
    Body:
        {
            "name": "Ma Watchlist",
            "tickers": ["AAPL", "GOOGL"],
            "description": "Description optionnelle"
        }
    """
    try:
        data = request.json
        
        if not data.get('name') or not data.get('tickers'):
            return jsonify({'error': 'Name et tickers requis'}), 400
        
        # Charger watchlists existantes
        watchlists = load_watchlists()
        
        # Générer ID unique
        import time
        new_id = f"custom_{int(time.time())}"
        
        # Créer nouvelle watchlist
        new_watchlist = {
            'id': new_id,
            'name': data['name'],
            'tickers': data['tickers'],
            'description': data.get('description', ''),
            'editable': True,
            'created_at': time.time()
        }
        
        watchlists.append(new_watchlist)
        
        # Sauvegarder
        if save_custom_watchlists(watchlists):
            logger.info(f"✅ Watchlist créée: {new_watchlist['name']}")
            return jsonify(new_watchlist), 201
        else:
            return jsonify({'error': 'Erreur sauvegarde'}), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur create watchlist: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_watchlists_bp.route('/watchlists/<watchlist_id>', methods=['DELETE'])
def delete_watchlist(watchlist_id):
    """
    Supprime une watchlist personnalisée
    """
    try:
        watchlists = load_watchlists()
        
        # Vérifier si watchlist existe et est éditable
        watchlist = next((wl for wl in watchlists if wl['id'] == watchlist_id), None)
        
        if not watchlist:
            return jsonify({'error': 'Watchlist non trouvée'}), 404
        
        if not watchlist.get('editable', True):
            return jsonify({'error': 'Impossible de supprimer une watchlist prédéfinie'}), 403
        
        # Supprimer
        watchlists = [wl for wl in watchlists if wl['id'] != watchlist_id]
        
        if save_custom_watchlists(watchlists):
            logger.info(f"✅ Watchlist supprimée: {watchlist_id}")
            return jsonify({'success': True, 'id': watchlist_id})
        else:
            return jsonify({'error': 'Erreur sauvegarde'}), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur delete watchlist: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@live_watchlists_bp.route('/watchlists/<watchlist_id>', methods=['PUT'])
def update_watchlist(watchlist_id):
    """
    Met à jour une watchlist personnalisée
    """
    try:
        data = request.json
        watchlists = load_watchlists()
        
        # Trouver watchlist
        watchlist = next((wl for wl in watchlists if wl['id'] == watchlist_id), None)
        
        if not watchlist:
            return jsonify({'error': 'Watchlist non trouvée'}), 404
        
        if not watchlist.get('editable', True):
            return jsonify({'error': 'Impossible de modifier une watchlist prédéfinie'}), 403
        
        # Mettre à jour
        if 'name' in data:
            watchlist['name'] = data['name']
        if 'tickers' in data:
            watchlist['tickers'] = data['tickers']
        if 'description' in data:
            watchlist['description'] = data['description']
        
        if save_custom_watchlists(watchlists):
            logger.info(f"✅ Watchlist mise à jour: {watchlist_id}")
            return jsonify(watchlist)
        else:
            return jsonify({'error': 'Erreur sauvegarde'}), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur update watchlist: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
