# main.py (VERSION V28 - BRAIN FACTORY)
# ---------------------------------------------------------
import os
import sys
import time
import subprocess
from config import FILE_MODEL_BRAIN

def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("🦅 ========================================== 🦅")
    print("      PLOUTOS TRADING SYSTEM - V28          ")
    print("      Station M1 Pro Max Edition            ")
    print("🦅 ========================================== 🦅")

def menu():
    while True:
        clear_screen()
        print_header()
        print("\n1. 📊 Dashboard (Web)")
        print("2. 🤖 Auto-Trader (Pilote)")
        print("3. 🌙 Scan de Nuit")
        print("\n--- ENTRAÎNEMENT IA ---")
        print("4. 🧠 Mode Sniper (1 Cerveau Généraliste)")
        print("5. 🏭 Mode Usine (4 Cerveaux Spécialisés en Parallèle)")
        print("\n--- OUTILS ---")
        print("6. ⏳ Backtester Visuel")
        print("7. 🧹 Vider Cache")
        print("0. Quitter")
        
        choix = input("\nChoix : ")
        
        if choix == "1":
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
            
        elif choix == "2":
            try: subprocess.run([sys.executable, "auto_trader.py"])
            except KeyboardInterrupt: pass
                
        elif choix == "3":
            subprocess.run([sys.executable, "night_scan.py"])
            input("Fin scan...")

        elif choix == "4":
            print("🧠 Lancement Entraînement Unique...")
            subprocess.run([sys.executable, "ai_trainer.py"])
            input("Terminé.")

        elif choix == "5":
            print("🏭 DÉMARRAGE DE L'USINE À CERVEAUX...")
            print("🚀 Lancement de 4 processus en parallèle (Surcharge CPU autorisée)")
            
            sectors = ["TECH", "DEFENSIVE", "ENERGY", "CRYPTO"]
            procs = []
            
            for s in sectors:
                print(f"   -> Lancement Cerveau {s}...")
                # On lance en background (Popen) sans attendre
                p = subprocess.Popen([sys.executable, "ai_trainer.py", s])
                procs.append(p)
            
            print("\n✅ Les 4 entraînements tournent en fond.")
            print("Ouvrez votre Moniteur d'Activité pour voir le spectacle.")
            print("Appuyez sur Entrée quand ils auront tous fini (logs dans terminaux).")
            
            # On attend qu'ils finissent tous
            for p in procs:
                p.wait()
            
            print("\n✅ TOUS LES CERVEAUX SONT PRÊTS.")
            input()

        elif choix == "6":
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "backtest_visual.py"])

        elif choix == "7":
            import shutil
            if os.path.exists("cache_data"): shutil.rmtree("cache_data")
            print("Cache vidé."); time.sleep(1)
            
        elif choix == "0": sys.exit()

if __name__ == "__main__":
    menu()
