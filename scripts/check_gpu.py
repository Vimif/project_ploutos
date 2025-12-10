#!/usr/bin/env python3
"""🔍 DIAGNOSTIC GPU CUDA

Vérifie pourquoi le GPU n'est pas détecté
"""

import sys
import subprocess
import os

print("\n" + "="*70)
print("🔍 DIAGNOSTIC GPU CUDA")
print("="*70 + "\n")

# 1. Vérifier nvidia-smi
print("1️⃣  Vérification nvidia-smi...")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ nvidia-smi fonctionne\n")
        print(result.stdout)
    else:
        print("❌ nvidia-smi ne fonctionne pas")
        print(f"Erreur: {result.stderr}")
except FileNotFoundError:
    print("❌ nvidia-smi non trouvé")
    print("\n💡 Solution: Installer NVIDIA drivers")
    print("   sudo apt update")
    print("   sudo apt install nvidia-driver-535")
    sys.exit(1)

print("\n" + "-"*70 + "\n")

# 2. Vérifier CUDA
print("2️⃣  Vérification CUDA...")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ CUDA installé\n")
        print(result.stdout)
    else:
        print("⚠️  CUDA pas détecté (mais pas nécessaire si PyTorch a CUDA built-in)")
except FileNotFoundError:
    print("⚠️  nvcc non trouvé (normal si PyTorch pré-compilé avec CUDA)")

print("\n" + "-"*70 + "\n")

# 3. Vérifier variables d'environnement
print("3️⃣  Variables d'environnement CUDA...")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']
for var in cuda_vars:
    value = os.environ.get(var)
    if value:
        print(f"  ✅ {var}={value}")
    else:
        print(f"  ⚠️  {var} non défini")

print("\n" + "-"*70 + "\n")

# 4. Vérifier PyTorch
print("4️⃣  Vérification PyTorch...")
try:
    import torch
    print(f"✅ PyTorch installé: version {torch.__version__}")
    print(f"  • CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  • CUDA version: {torch.version.cuda}")
        print(f"  • GPU count: {torch.cuda.device_count()}")
        print(f"  • Current device: {torch.cuda.current_device()}")
        print(f"  • Device name: {torch.cuda.get_device_name(0)}")
        print(f"  • Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  • Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Test simple
        print("\n  🧪 Test rapide GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  ✅ Test GPU réussi !")
        
    else:
        print("\n  ❌ PyTorch ne détecte pas CUDA !")
        print("\n  🔍 Causes possibles:")
        print("    1. PyTorch installé sans support CUDA")
        print("    2. Version CUDA incompatible avec PyTorch")
        print("    3. Driver NVIDIA incompatible")
        
        print("\n  💡 Solutions:")
        print("\n    Option 1: Réinstaller PyTorch avec CUDA")
        print("    -----------------------------------------")
        print("    pip3 uninstall torch torchvision torchaudio")
        print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        print("\n    Option 2: Vérifier compatibilité")
        print("    ------------------------------------")
        print("    Aller sur: https://pytorch.org/get-started/locally/")
        print("    Sélectionner votre config et copier la commande")
        
except ImportError:
    print("❌ PyTorch non installé")
    print("\n💡 Installation:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "-"*70 + "\n")

# 5. Vérifier version driver vs CUDA
print("5️⃣  Compatibilité Driver <→ CUDA...")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        driver_version = result.stdout.strip()
        print(f"  • Driver NVIDIA: {driver_version}")
        
        # Extraire version majeure
        try:
            driver_major = int(driver_version.split('.')[0])
            
            if driver_major >= 535:
                print("  ✅ Driver compatible CUDA 12.x")
            elif driver_major >= 520:
                print("  ✅ Driver compatible CUDA 11.8")
            elif driver_major >= 470:
                print("  ⚠️  Driver ancien, compatible CUDA 11.4")
            else:
                print("  ❌ Driver trop ancien, mise à jour recommandée")
                
        except:
            pass
except:
    print("  ⚠️  Impossible de vérifier version driver")

print("\n" + "="*70)
print("🎉 DIAGNOSTIC TERMINÉ")
print("="*70 + "\n")

# Résumé
print("📊 RÉSUMÉ:\n")

try:
    import torch
    if torch.cuda.is_available():
        print("✅ GPU FONCTIONNEL - Prêt pour entraînement !")
        print(f"\n   Utiliser: bash scripts/train_v4_optimal.sh")
    else:
        print("❌ GPU NON DÉTECTÉ par PyTorch")
        print("\n   🔧 Actions à faire:")
        print("   1. Vérifier que nvidia-smi fonctionne")
        print("   2. Réinstaller PyTorch avec CUDA")
        print("   3. Redémarrer terminal/session")
        print("\n   En attendant, utiliser config CPU:")
        print("   bash scripts/train_v4_optimal.sh --config config/training_config_v4_optimal_cpu.yaml")
except:
    print("❌ PyTorch non installé")

print("\n")
