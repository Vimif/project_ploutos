// Connexion WebSocket
const socket = io();

// État de connexion
socket.on('connect', () => {
    console.log('✅ Connecté au serveur');
    loadAllData();
});

socket.on('disconnect', () => {
    console.log('❌ Déconnecté du serveur');
});

// Mises à jour temps réel
socket.on('account_update', (data) => {
    updateAccountStats(data);
});

// Formatage
function formatMoney(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function loadAllData() {
    loadAccount();
    loadPositions();
    loadPerformance();
}

// Charger compte
async function loadAccount() {
    try {
        const response = await fetch('/api/account');
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('portfolio').textContent = formatMoney(result.data.portfolio_value);
            document.getElementById('cash').textContent = formatMoney(result.data.cash);
        }
    } catch (error) {
        console.error('❌ Erreur account:', error);
        document.getElementById('portfolio').textContent = 'Erreur';
    }
}

// Charger positions
async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        const result = await response.json();
        
        if (result.success) {
            const tbody = document.getElementById('positions-body');
            
            if (result.data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center">Aucune position</td></tr>';
            } else {
                tbody.innerHTML = result.data.map(p => `
                    <tr>
                        <td><strong>${p.symbol}</strong></td>
                        <td>${p.qty}</td>
                        <td>${formatMoney(p.current_price)}</td>
                        <td>${formatMoney(p.market_value)}</td>
                        <td class="${p.unrealized_pl >= 0 ? 'positive' : 'negative'}">
                            ${formatMoney(p.unrealized_pl)} (${p.unrealized_plpc.toFixed(2)}%)
                        </td>
                    </tr>
                `).join('');
            }
        }
    } catch (error) {
        console.error('❌ Erreur positions:', error);
    }
}

// Charger performance
async function loadPerformance() {
    try {
        const response = await fetch('/api/performance');
        const result = await response.json();
        
        if (result.success) {
            const plEl = document.getElementById('pl');
            const pl = result.data.total_unrealized_pl;
            plEl.textContent = formatMoney(pl);
            plEl.className = pl >= 0 ? 'positive' : 'negative';
            
            if (document.getElementById('total-pos')) {
                document.getElementById('total-pos').textContent = result.data.total_positions;
                document.getElementById('win-pos').textContent = result.data.winning_positions;
                document.getElementById('lose-pos').textContent = result.data.losing_positions;
                document.getElementById('win-rate').textContent = result.data.win_rate.toFixed(1) + '%';
            }
        }
    } catch (error) {
        console.error('❌ Erreur performance:', error);
    }
}

// Charger au démarrage
loadAllData();

// Rafraîchir toutes les 10s
setInterval(loadAllData, 10000);