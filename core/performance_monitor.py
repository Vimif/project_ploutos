#!/usr/bin/env python3
"""
Performance Monitor Callback
Surveille FPS, GPU, CPU en temps réel et détecte bottlenecks
"""

import time
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except:
    HAS_NVML = False

class PerformanceMonitor(BaseCallback):
    """
    Surveille performance et détecte bottlenecks
    """
    
    def __init__(self, log_freq=5000, verbose=1):
        """
        Args:
            log_freq: Fréquence de log (en steps)
            verbose: Niveau de verbosité
        """
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.last_log_step = 0
        
        # Buffers
        self.fps_buffer = deque(maxlen=100)
        self.step_times = deque(maxlen=1000)
        
        # Timers
        self.last_time = time.time()
        self.last_step = 0
        
        # GPU handle
        self.gpu_handle = None
        if HAS_NVML:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass
    
    def _on_training_start(self):
        if self.verbose > 0:
            print("\n📊 Performance Monitor activé")
            print(f"   Log freq : {self.log_freq:,} steps")
            print(f"   GPU      : {'Détecté' if self.gpu_handle else 'Non disponible'}")
            print(f"   CPU      : {'Détecté' if HAS_PSUTIL else 'Non disponible'}\n")
        
        self.last_time = time.time()
        self.last_step = self.num_timesteps
    
    def _on_step(self) -> bool:
        """
        Calcule FPS et log périodiquement
        """
        current_time = time.time()
        
        # Calculer FPS instantané
        steps_done = self.num_timesteps - self.last_step
        time_elapsed = current_time - self.last_time
        
        if time_elapsed > 0:
            fps = steps_done / time_elapsed
            self.fps_buffer.append(fps)
            self.step_times.append(time_elapsed / steps_done if steps_done > 0 else 0)
        
        # Log périodique
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_performance()
            self.last_log_step = self.num_timesteps
        
        # Update timers
        self.last_time = current_time
        self.last_step = self.num_timesteps
        
        return True
    
    def _log_performance(self):
        """
        Log métriques de performance
        """
        metrics = {}
        
        # FPS
        if len(self.fps_buffer) > 0:
            avg_fps = np.mean(list(self.fps_buffer))
            metrics['perf/fps'] = avg_fps
            
            # Détection bottleneck FPS
            if avg_fps < 10000:
                status = "❌ TRÈS LENT"
            elif avg_fps < 20000:
                status = "⚠️  LENT"
            elif avg_fps < 30000:
                status = "🟡 OK"
            else:
                status = "✅ RAPIDE"
            
            if self.verbose > 0:
                print(f"\n⏱️  Performance (step {self.num_timesteps:,})")
                print(f"   FPS : {avg_fps:,.0f} {status}")
        
        # GPU
        if self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_util = util.gpu
                mem_used_gb = mem_info.used / 1024**3
                mem_total_gb = mem_info.total / 1024**3
                
                metrics['perf/gpu_util'] = gpu_util
                metrics['perf/gpu_memory_gb'] = mem_used_gb
                metrics['perf/gpu_temp'] = temp
                
                # Détection bottleneck GPU
                if gpu_util < 30:
                    gpu_status = "❌ SOUS-UTILISÉ (CPU bottleneck?)"
                elif gpu_util < 60:
                    gpu_status = "⚠️  Peut mieux faire"
                else:
                    gpu_status = "✅ Bien utilisé"
                
                if self.verbose > 0:
                    print(f"   GPU : {gpu_util}% {gpu_status}")
                    print(f"   VRAM: {mem_used_gb:.1f}/{mem_total_gb:.1f} GB")
                    print(f"   Temp: {temp}°C")
            except:
                pass
        
        # CPU
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['perf/cpu_util'] = cpu_percent
            
            if self.verbose > 0:
                if cpu_percent > 90:
                    cpu_status = "❌ SATURÉ (bottleneck!)"
                elif cpu_percent > 70:
                    cpu_status = "⚠️  Chargé"
                else:
                    cpu_status = "✅ OK"
                
                print(f"   CPU : {cpu_percent:.0f}% {cpu_status}")
        
        # Step time
        if len(self.step_times) > 0:
            avg_step_time = np.mean(list(self.step_times)) * 1000  # ms
            metrics['perf/step_time_ms'] = avg_step_time
            
            if self.verbose > 0:
                print(f"   Step: {avg_step_time:.2f}ms")
        
        # Logger dans W&B
        if wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)
        
        # Diagnostics
        if self.verbose > 0:
            self._print_diagnostics(metrics)
    
    def _print_diagnostics(self, metrics):
        """
        Affiche diagnostics si problèmes détectés
        """
        fps = metrics.get('perf/fps', 0)
        gpu_util = metrics.get('perf/gpu_util', 100)
        cpu_util = metrics.get('perf/cpu_util', 0)
        
        # Bottleneck CPU
        if cpu_util > 90 and gpu_util < 50:
            print("\n⚠️  BOTTLENECK CPU DÉTECTÉ")
            print("   Solutions:")
            print("   1. Réduire n_envs (16→8→4)")
            print("   2. Utiliser DummyVecEnv au lieu de SubprocVecEnv")
            print("   3. Augmenter batch_size pour compenser\n")
        
        # GPU sous-utilisé
        elif gpu_util < 30 and fps < 20000:
            print("\n⚠️  GPU SOUS-UTILISÉ")
            print("   Causes possibles:")
            print("   1. Batch size trop petit")
            print("   2. CPU bottleneck (vérifier ci-dessus)")
            print("   3. I/O disk lent (peu probable avec numpy)\n")
        
        # Tout va bien
        elif fps > 25000 and gpu_util > 60:
            print("\n✅ PERFORMANCE OPTIMALE")
            print(f"   FPS: {fps:,.0f}")
            print(f"   GPU: {gpu_util}%\n")
    
    def _on_training_end(self):
        if self.verbose > 0:
            print("\n✅ Performance Monitor terminé")
            
            if len(self.fps_buffer) > 0:
                avg_fps = np.mean(list(self.fps_buffer))
                print(f"   FPS moyen : {avg_fps:,.0f}")
        
        # Cleanup GPU
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

if __name__ == '__main__':
    print("\n" + "="*80)
    print("📊 PERFORMANCE MONITOR CALLBACK")
    print("="*80 + "\n")
    
    print("📝 Usage dans train_curriculum.py :\n")
    print("""from core.performance_monitor import PerformanceMonitor

# Créer callback
perf_monitor = PerformanceMonitor(
    log_freq=5000,  # Log toutes les 5k steps
    verbose=1
)

# Combiner avec autres callbacks
callback = CallbackList([
    checkpoint_callback,
    wandb_callback,
    trading_callback,
    perf_monitor  # ✅ Ajouté
])
""")
    
    print("\n📊 Métriques loggées :")
    print("   perf/fps")
    print("   perf/gpu_util")
    print("   perf/gpu_memory_gb")
    print("   perf/gpu_temp")
    print("   perf/cpu_util")
    print("   perf/step_time_ms")
    print("\n")
