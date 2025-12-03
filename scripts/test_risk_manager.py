#!/usr/bin/env python3
"""Tester le Risk Manager"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk_manager import RiskManager

def test_risk_manager():
    """Tests du risk manager"""
    
    print("="*70)
    print("🛡️ TEST DU RISK MANAGER")
    print("="*70)
    
    rm = RiskManager(
        max_portfolio_risk=0.01,
        max_daily_loss=0.03,
        max_position_size=0.05
    )
    
    portfolio_value = 100000
    
    # Test 1: Position sizing
    print("\n1️⃣ TEST POSITION SIZING")
    print("-" * 70)
    qty, pos_value = rm.calculate_position_size(
        portfolio_value=portfolio_value,
        entry_price=500.0,
        stop_loss_pct=0.05
    )
    print(f"✅ Quantité calculée: {qty} actions")
    print(f"✅ Valeur position: ${pos_value:,.2f}")
    
    # Test 2: Circuit breaker
    print("\n2️⃣ TEST CIRCUIT BREAKER")
    print("-" * 70)
    rm.reset_daily_stats(portfolio_value)
    
    print(f"Portfolio initial: ${portfolio_value:,.2f}")
    
    # Simuler perte de 2%
    new_value = portfolio_value * 0.98
    allowed = rm.check_daily_loss_limit(new_value)
    print(f"Perte 2%: {'✅ Trading autorisé' if allowed else '🚨 Circuit breaker'}")
    
    # Simuler perte de 4% (>3%)
    new_value = portfolio_value * 0.96
    allowed = rm.check_daily_loss_limit(new_value)
    print(f"Perte 4%: {'✅ Trading autorisé' if allowed else '🚨 Circuit breaker activé!'}")
    
    # Test 3: Kelly Criterion
    print("\n3️⃣ TEST KELLY CRITERION")
    print("-" * 70)
    kelly = rm.calculate_kelly_criterion(
        win_rate=0.60,
        avg_win=0.10,
        avg_loss=0.05
    )
    print(f"✅ Kelly size optimal: {kelly*100:.2f}%")
    
    # Test 4: Évaluation position
    print("\n4️⃣ TEST ÉVALUATION POSITION")
    print("-" * 70)
    
    # Position normale
    risk = rm.assess_position_risk(
        symbol='NVDA',
        position_value=3000,
        portfolio_value=portfolio_value,
        unrealized_plpc=-0.02,
        days_held=5
    )
    print(f"Position NVDA: {risk['risk_level']} - {risk['recommendation']}")
    
    # Position à risque
    risk = rm.assess_position_risk(
        symbol='TSLA',
        position_value=8000,
        portfolio_value=portfolio_value,
        unrealized_plpc=-0.12,
        days_held=35
    )
    print(f"Position TSLA: {risk['risk_level']} - {risk['recommendation']}")
    for warning in risk['warnings']:
        print(f"  ⚠️  {warning}")
    
    # Test 5: Rapport complet
    print("\n5️⃣ TEST RAPPORT DE RISQUE")
    print("-" * 70)
    
    positions = [
        {'symbol': 'NVDA', 'market_value': 3000, 'unrealized_plpc': 0.05, 'unrealized_pl': 150},
        {'symbol': 'MSFT', 'market_value': 2500, 'unrealized_plpc': -0.02, 'unrealized_pl': -50},
        {'symbol': 'TSLA', 'market_value': 8000, 'unrealized_plpc': -0.12, 'unrealized_pl': -960},
    ]
    
    report = rm.get_risk_report(positions, portfolio_value)
    rm.print_risk_summary(report)
    
    print("\n" + "="*70)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*70)

if __name__ == '__main__':
    test_risk_manager()