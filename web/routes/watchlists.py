#!/usr/bin/env python3
"""
📊 WATCHLISTS API - Serve predefined stock lists
"""

import json
from pathlib import Path
from flask import Blueprint, jsonify

watchlists_bp = Blueprint('watchlists', __name__)

# Load watchlists from config
WATCHLISTS_FILE = Path(__file__).parent.parent.parent / 'config' / 'watchlists.json'

def load_watchlists():
    """Load watchlists from JSON file"""
    try:
        with open(WATCHLISTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erreur chargement watchlists: {e}")
        return {}

# Cache global
WATCHLISTS = load_watchlists()

@watchlists_bp.route('/api/watchlists')
def get_all_watchlists():
    """
    Retourne toutes les watchlists avec métadonnées
    """
    return jsonify({
        'count': len(WATCHLISTS),
        'watchlists': WATCHLISTS
    })

@watchlists_bp.route('/api/watchlists/<list_id>')
def get_watchlist(list_id):
    """
    Retourne une watchlist spécifique
    """
    if list_id not in WATCHLISTS:
        return jsonify({'error': f'Watchlist {list_id} non trouvée'}), 404
    
    return jsonify(WATCHLISTS[list_id])

@watchlists_bp.route('/api/watchlists/<list_id>/tickers')
def get_watchlist_tickers(list_id):
    """
    Retourne uniquement les tickers d'une watchlist
    """
    if list_id not in WATCHLISTS:
        return jsonify({'error': f'Watchlist {list_id} non trouvée'}), 404
    
    return jsonify({
        'list_id': list_id,
        'name': WATCHLISTS[list_id]['name'],
        'tickers': WATCHLISTS[list_id]['tickers']
    })

@watchlists_bp.route('/api/watchlists/categories')
def get_categories():
    """
    Retourne les catégories de watchlists (résumé)
    """
    categories = {}
    
    for list_id, data in WATCHLISTS.items():
        # Déterminer la catégorie depuis le nom
        if '🇺🇸' in data['name'] or 'US' in data['name']:
            cat = 'us'
        elif '🇫🇷' in data['name'] or 'France' in data['name']:
            cat = 'fr'
        elif '🇨🇳' in data['name'] or 'China' in data['name']:
            cat = 'international'
        elif 'ETF' in data['name']:
            cat = 'etfs'
        elif 'Crypto' in data['name']:
            cat = 'crypto'
        else:
            cat = 'other'
        
        if cat not in categories:
            categories[cat] = []
        
        categories[cat].append({
            'id': list_id,
            'name': data['name'],
            'description': data['description'],
            'count': len(data['tickers'])
        })
    
    return jsonify(categories)
