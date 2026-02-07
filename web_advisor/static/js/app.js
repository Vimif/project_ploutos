/**
 * Ploutos Advisory - Logique applicative
 *
 * Utilitaires partages entre les pages.
 */

// Formattage monnaie
function formatMoney(value) {
    return new Intl.NumberFormat('fr-FR', {
        style: 'currency',
        currency: 'USD',
    }).format(value);
}

// Formattage pourcentage
function formatPercent(value) {
    return new Intl.NumberFormat('fr-FR', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
    }).format(value);
}
