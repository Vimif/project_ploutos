// Connexion WebSocket
const socket = io();

// État de connexion
socket.on('connect', () => {
    console.log('✅ Connecté au serveur');
    updateStatus(true);
    loadAllData();
});

socket.on('disconnect', () => {
    console.log('❌ Déconnecté du serveur');
    updateStatus(false);
});

function updateStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    
    if (connected) {
        dot.style.background = '#10b981';
        text.textContent = 'Connecté';
    } else {
        dot.style.background = '#ef4444';
        text.textContent = 'Déconnecté';
    }
}

function formatMoney(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function loadAllData() {
    loadAccount();
    loadPositions();
    loadOrders();
    loadPerformance();
}

async function loadAccount() {
    try {
        const response = await fetch('/api/account');
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('portfolio').textContent = formatMoney(result.data.portfolio_value);
            document.getElementById('cash').textContent = formatMoney(result.data.cash);
            document.getElementById('buying-power').textContent = formatMoney(result.data.buying_power);
            
            const dailyPL = result.data.equity - result.data.last_equity;
            const dailyPLPct = (dailyPL / result.data.last_equity * 100).toFixed(2);
            const dailyPLElement = document.getElementById('daily-pl');
            
            dailyPLElement.textContent = `${formatMoney(dailyPL)} (${dailyPLPct}%)`;
            dailyPLElement.className = 'stat-change ' + (dailyPL >= 0 ? 'positive' : 'negative');
        }
    } catch (error) {
        console.error('❌ Erreur account:', error);
    }
}

async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        const result = await response.json();
        
        if (result.success) {
            const tbody = document.getElementById('positions-body');
            
            if (result.data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="loading">Aucune position ouverte</td></tr>';
            } else {
                tbody.innerHTML = result.data.map(p => `
                    <tr>
                        <td><strong>${p.symbol}</strong></td>
                        <td>${p.qty}</td>
                        <td>${formatMoney(p.avg_entry_price)}</td>
                        <td>${formatMoney(p.current_price)}</td>
                        <td>${formatMoney(p.market_value)}</td>
                        <td class="${p.unrealized_pl >= 0 ? 'positive' : 'negative'}">
                            ${formatMoney(p.unrealized_pl)}
                        </td>
                        <td class="${p.unrealized_plpc >= 0 ? 'positive' : 'negative'}">
                            ${p.unrealized_plpc.toFixed(2)}%
                        </td>
                        <td>
                            <button class="btn-close" onclick="closePosition('${p.symbol}')">
                                Fermer
                            </button>
                        </td>
                    </tr>
                `).join('');
            }
        }
    } catch (error) {
        console.error('❌ Erreur positions:', error);
    }
}

async function loadOrders() {
    try {
        const response = await fetch('/api/orders');
        const result = await response.json();
        
        if (result.success) {
            const tbody = document.getElementById('orders-body');
            
            if (result.data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="loading">Aucun ordre récent</td></tr>';
            } else {
                tbody.innerHTML = result.data.slice(0, 20).map(order => `
                    <tr>
                        <td>${formatDate(order.filled_at)}</td>
                        <td><strong>${order.symbol}</strong></td>
                        <td style="color: ${order.side === 'buy' ? '#10b981' : '#ef4444'}">
                            ${order.side.toUpperCase()}
                        </td>
                        <td>${order.qty}</td>
                        <td>${formatMoney(order.filled_avg_price)}</td>
                        <td>${order.status}</td>
                    </tr>
                `).join('');
            }
        }
    } catch (error) {
        console.error('❌ Erreur orders:', error);
    }
}

async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const result = await response.json();
        
        if (result.success) {
            const plEl = document.getElementById('pl');
            const pl = result.data.total_unrealized_pl;
            plEl.textContent = formatMoney(pl);
            
            const plPctElement = document.getElementById('pl-pct');
            plPctElement.textContent = `${result.data.total_unrealized_plpc.toFixed(2)}%`;
            plPctElement.className = 'stat-change ' + (pl >= 0 ? 'positive' : 'negative');
            
            document.getElementById('total-pos').textContent = result.data.total_positions;
            document.getElementById('win-pos').textContent = result.data.winning_positions;
            document.getElementById('lose-pos').textContent = result.data.losing_positions;
            document.getElementById('win-rate').textContent = `${result.data.win_rate.toFixed(1)}%`;
        }
    } catch (error) {
        console.error('❌ Erreur performance:', error);
    }
}

async function closePosition(symbol) {
    if (!confirm(`Voulez-vous vraiment fermer la position ${symbol} ?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/close_position/${symbol}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(`✅ Position ${symbol} fermée`);
            loadAllData();
        } else {
            alert(`❌ Erreur: ${result.error}`);
        }
    } catch (error) {
        console.error('❌ Erreur closePosition:', error);
        alert('❌ Erreur lors de la fermeture');
    }
}

// Charger au démarrage
loadAllData();

// Rafraîchir toutes les 10 secondes
setInterval(loadAllData, 10000);