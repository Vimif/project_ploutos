# project_ploutos/__init__.py
"""
Package Ploutos - Système de trading multi-cerveaux
"""
import sys
from pathlib import Path

# Ajouter le projet au path si nécessaire
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

__version__ = "2.0.0"
