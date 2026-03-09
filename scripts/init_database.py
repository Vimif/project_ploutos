#!/usr/bin/env python3
"""Initialiser la base de données Ploutos"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import init_database

if __name__ == '__main__':
    print("="*70)
    print("🗄️  INITIALISATION BASE DE DONNÉES PLOUTOS")
    print("="*70)
    
    try:
        init_database()
        print("\n✅ Base de données initialisée avec succès!")
        print("\nTables créées:")
        print("  - trades")
        print("  - positions")
        print("  - daily_summary")
        print("  - predictions")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)