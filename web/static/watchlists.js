/**
 * 📊 WATCHLISTS MODULE
 * Gère l'affichage et l'interaction avec les watchlists dans la sidebar
 */

class WatchlistsManager {
    constructor() {
        this.watchlists = {};
        this.categories = {};
        this.sidebarVisible = true;
        this.init();
    }
    
    async init() {
        console.log('📊 Initialisation Watchlists Manager...');
        
        // 🔄 Charger les watchlists
        await this.loadWatchlists();
        
        // 🎯 Event listeners
        this.setupEventListeners();
        
        // 📊 Afficher les watchlists
        this.renderWatchlists();
    }
    
    async loadWatchlists() {
        try {
            const response = await fetch('/api/watchlists');
            if (!response.ok) throw new Error('Erreur chargement watchlists');
            
            const data = await response.json();
            this.watchlists = data.watchlists;
            
            console.log(`✅ ${data.count} watchlists chargées`);
            
            // Charger les catégories
            const catResponse = await fetch('/api/watchlists/categories');
            if (catResponse.ok) {
                this.categories = await catResponse.json();
            }
            
        } catch (error) {
            console.error('❌ Erreur chargement watchlists:', error);
            this.renderError();
        }
    }
    
    setupEventListeners() {
        // Toggle sidebar
        const toggleBtn = document.getElementById('toggle-sidebar');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleSidebar());
        }
        
        // Search
        const searchInput = document.getElementById('watchlist-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.filterWatchlists(e.target.value));
        }
    }
    
    toggleSidebar() {
        this.sidebarVisible = !this.sidebarVisible;
        const sidebar = document.getElementById('watchlist-sidebar');
        const mainContent = document.getElementById('main-content');
        
        if (this.sidebarVisible) {
            sidebar.classList.remove('hidden');
            mainContent.style.marginLeft = '280px';
        } else {
            sidebar.classList.add('hidden');
            mainContent.style.marginLeft = '0';
        }
    }
    
    renderWatchlists() {
        const container = document.getElementById('watchlists-container');
        if (!container) return;
        
        let html = '';
        
        // Grouper par catégories si disponible
        if (Object.keys(this.categories).length > 0) {
            for (const [catId, lists] of Object.entries(this.categories)) {
                html += this.renderCategory(catId, lists);
            }
        } else {
            // Affichage simple sans catégories
            for (const [listId, listData] of Object.entries(this.watchlists)) {
                html += this.renderWatchlistItem(listId, listData);
            }
        }
        
        container.innerHTML = html;
        
        // 🎯 Ajouter les event listeners
        this.attachClickListeners();
    }
    
    renderCategory(catId, lists) {
        const catNames = {
            'us': '🇺🇸 Actions US',
            'fr': '🇫🇷 Actions FR',
            'international': '🌍 International',
            'etfs': '📈 ETFs',
            'crypto': '₿ Crypto',
            'other': '📂 Autres'
        };
        
        let html = `
            <div class="mb-4">
                <h4 class="text-xs font-bold text-gray-500 uppercase mb-2 px-2">${catNames[catId] || catId}</h4>
                <div class="space-y-1">
        `;
        
        for (const list of lists) {
            const fullData = this.watchlists[list.id];
            if (fullData) {
                html += this.renderWatchlistItem(list.id, fullData, true);
            }
        }
        
        html += `
                </div>
            </div>
        `;
        
        return html;
    }
    
    renderWatchlistItem(listId, listData, compact = false) {
        const tickerCount = listData.tickers.length;
        const market = listId.includes('_fr') || listId.includes('cac40') || listId.includes('luxury') ? 'fr' : 'us';
        
        return `
            <div class="watchlist-item bg-gray-700 rounded px-3 py-2 hover:bg-gray-600" 
                 data-list-id="${listId}">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <div class="text-sm font-semibold">${listData.name}</div>
                        ${!compact ? `<div class="text-xs text-gray-400 mt-1">${listData.description}</div>` : ''}
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="ticker-badge ${market}">${tickerCount}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    attachClickListeners() {
        const items = document.querySelectorAll('.watchlist-item');
        items.forEach(item => {
            item.addEventListener('click', () => {
                const listId = item.dataset.listId;
                this.selectWatchlist(listId);
            });
        });
    }
    
    async selectWatchlist(listId) {
        console.log(`🎯 Sélection watchlist: ${listId}`);
        
        try {
            const response = await fetch(`/api/watchlists/${listId}/tickers`);
            if (!response.ok) throw new Error('Erreur chargement tickers');
            
            const data = await response.json();
            
            console.log(`✅ ${data.tickers.length} tickers:`, data.tickers);
            
            // 📢 Émettre un événement pour que d'autres modules puissent réagir
            window.dispatchEvent(new CustomEvent('watchlistSelected', {
                detail: {
                    listId: listId,
                    name: data.name,
                    tickers: data.tickers
                }
            }));
            
            // 🔔 Notification visuelle
            this.showNotification(`✅ ${data.name} chargée (${data.tickers.length} tickers)`);
            
        } catch (error) {
            console.error('❌ Erreur sélection watchlist:', error);
            this.showNotification('❌ Erreur chargement watchlist', 'error');
        }
    }
    
    filterWatchlists(query) {
        const items = document.querySelectorAll('.watchlist-item');
        const lowerQuery = query.toLowerCase();
        
        items.forEach(item => {
            const text = item.textContent.toLowerCase();
            if (text.includes(lowerQuery)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    }
    
    showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `fixed top-20 right-4 px-4 py-3 rounded shadow-lg z-50 ${
            type === 'error' ? 'bg-red-600' : 'bg-green-600'
        } text-white`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    renderError() {
        const container = document.getElementById('watchlists-container');
        if (!container) return;
        
        container.innerHTML = `
            <div class="text-center text-red-400 py-4">
                <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                <p class="text-xs">Erreur chargement watchlists</p>
            </div>
        `;
    }
}

// 🚀 Auto-init
if (typeof window.WatchlistsManager === 'undefined') {
    window.WatchlistsManager = new WatchlistsManager();
    console.log('✅ Watchlists Manager initialisé');
}

// 📡 Exemple d'écoute d'événements (pour d'autres modules)
window.addEventListener('watchlistSelected', (event) => {
    console.log('📡 Watchlist sélectionnée:', event.detail);
    // Ici, d'autres modules peuvent réagir (ex: lancer un screener)
});
