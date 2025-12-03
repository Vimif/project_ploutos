#!/usr/bin/env python3
# fix_imports_v2.py
"""Fix automatique de tous les imports"""

from pathlib import Path
import re

# Code à ajouter au début de chaque fichier
FIX_CODE = '''# Auto-fix imports
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

'''

# Fichiers à traiter
files_to_fix = [
    'config/settings.py',
    'config/tickers.py',
    'core/features.py',
    'core/environment.py',
    'core/models.py',
    'core/utils.py',
    'trading/brain_trader.py',
    'trading/portfolio.py',
    'training/trainer.py',
    'ui/dashboard.py',
    'scripts/train_models.py',
    'scripts/run_trader.py',
    'scripts/backtest.py',
]

def fix_file(filepath):
    """Ajouter le fix à un fichier"""
    path = Path(filepath)
    
    if not path.exists():
        print(f"⏭️  {filepath} (n'existe pas)")
        return
    
    # Lire le contenu
    content = path.read_text()
    
    # Vérifier si déjà fixé
    if 'Auto-fix imports' in content or 'FIX PATH' in content:
        print(f"⏭️  {filepath} (déjà fixé)")
        return
    
    # Trouver où commence vraiment le code (après docstring et commentaires)
    lines = content.split('\n')
    insert_pos = 0
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Gérer les docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:  # Docstring sur une ligne
                    in_docstring = False
                    insert_pos = i + 1
                continue
        else:
            if docstring_char in stripped:
                in_docstring = False
                insert_pos = i + 1
                continue
        
        # Ignorer les commentaires et lignes vides
        if not stripped or stripped.startswith('#'):
            continue
        
        # Première vraie ligne de code
        if not in_docstring:
            insert_pos = i
            break
    
    # Insérer le fix
    lines.insert(insert_pos, FIX_CODE)
    
    # Écrire
    path.write_text('\n'.join(lines))
    print(f"✅ {filepath}")

def main():
    print("🔧 Fix automatique des imports\n")
    
    for filepath in files_to_fix:
        fix_file(filepath)
    
    print("\n✅ Terminé")
    print("\nTestez maintenant:")
    print("  streamlit run ui/dashboard.py")

if __name__ == "__main__":
    main()
