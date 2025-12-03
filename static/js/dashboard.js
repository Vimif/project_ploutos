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

// Mises à jour temps réel
socket.on('account_update', (data) => {
    updateAccountStats(data);
});

socket.on('positions_update', (data) => {
    console.log('📊 Mise à jour positions:', data);
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

function loadAllData() {
    loadAccount();
    loadPositions();
    loadOrders();
    loadPerformance();
}

// Charger les données du compte
async function loadAccount() {
    try {
        const response = await fetch('/api/account');
        const result = await response.json();
        
        if (result.success) {
            updateAccountStats(result.data);
        }
    } catch (error) {
        console.error('❌ Erreur loadAccount:', error);
    }
}

function updateAccountStats(data) {
    document.getElementById('portfolio-value').textContent = formatMoney(data.portfolio_value);
    document.getElementById('cash-value').textContent = formatMoney(data.cash);
    document.getElementById('buying-power').textContent = formatMoney(data.buying_power);
    
    // P&L quotidien
    const dailyPL = data.equity - data.last_equity;
    const dailyPLPct = (dailyPL / data.last_equity * 100).toFixed(2);
    const dailyPLElement = document.getElementById('daily-pl');
    
    dailyPLElement.textContent = `${formatMoney(dailyPL)} (${dailyPLPct}%)`;
    dailyPLElement.className = 'stat-change ' + (dailyPL >= 0 ? 'positive' : 'negative');
}

// Charger les positions
async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        const result = await response.json();
        
        if (result.success) {
            displayPositions(result.data);
        }
    } catch (error) {
        console.error('❌ Erreur loadPositions:', error);
    }
}

function displayPositions(positions) {
    const tbody = document.getElementById('positions-body');
    
    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="loading">Aucune position ouverte</td></tr>';
        return;
    }
    
    tbody.innerHTML = positions.map(pos => `
        <tr>
            <td><strong>${pos.symbol}</strong></td>
            <td>${pos.qty}</td>
            <td>${formatMoney(pos.avg_entry_price)}</td>
            <td>${formatMoney(pos.current_price)}</td>
            <td>${formatMoney(pos.market_value)}</td>
            <td class="${pos.unrealized_pl >= 0 ? 'positive' : 'negative'}">
                ${formatMoney(pos.unrealized_pl)}
            </td>
            <td class="${pos.unrealized_plpc >= 0 ? 'positive' : 'negative'}">
                ${pos.unrealized_plpc.toFixed(2)}%
            </td>
            <td>
                <button class="btn btn-close" onclick="closePosition('${pos.symbol}')">
                    Fermer
                </button>
            </td>
        </tr>
    `).join('');
}

// Charger les ordres
async function loadOrders() {
    try {
        const response = await fetch('/api/orders');
        const result = await response.json();
        
        if (result.success) {
            displayOrders(result.data);
        }
    } catch (error) {
        console.error('❌ Erreur loadOrders:', error);
    }
}

function displayOrders(orders) {
    const tbody = document.getElementById('orders-body');
    
    if (orders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading">Aucun ordre récent</td></tr>';
        return;
    }
    
    tbody.innerHTML = orders.slice(0, 20).map(order => `
        <tr>
            <td>${formatDate(order.filled_at || order.created_at)}</td>
            <td><strong>${order.symbol}</strong></td>
            <td>
                <span class="badge badge-${order.side}">${order.side.toUpperCase()}</span>
            </td>
            <td>${order.qty}</td>
            <td>${formatMoney(order.filled_avg_price)}</td>
            <td>
                <span class="badge badge-filled">${order.status}</span>
            </td>
        </tr>
    `).join('');
}

// Charger performance
async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const result = await response.json();
        
        if (result.success) {
            updatePerformanceStats(result.data);
        }
    } catch (error) {
        console.error('❌ Erreur loadPerformance:', error);
    }
}

function updatePerformanceStats(data) {
    document.getElementById('total-pl').textContent = formatMoney(data.total_unrealized_pl);
    
    const plPctElement = document.getElementById('total-pl-pct');
    plPctElement.textContent = `${data.total_unrealized_plpc.toFixed(2)}%`;
    plPctElement.className = 'stat-change ' + (data.total_unrealized_pl >= 0 ? 'positive' : 'negative');
    
    document.getElementById('total-positions').textContent = data.total_positions;
    document.getElementById('winning-positions').textContent = data.winning_positions;
    document.getElementById('losing-positions').textContent = data.losing_positions;
    document.getElementById('win-rate').textContent = `${data.win_rate.toFixed(1)}%`;
}

// Fermer une position
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

// Formatage
function formatMoney(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
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

// Rafraîchir toutes les 10 secondes
setInterval(loadAllData, 10000);