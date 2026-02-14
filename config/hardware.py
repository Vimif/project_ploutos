# config/hardware.py
"""Auto-détection hardware et calcul des paramètres optimaux.

Détecte GPU, CPU, RAM et calcule n_envs, batch_size, n_steps optimaux.
Utilisé par les scripts avec le flag --auto-scale.

Usage:
    from config.hardware import auto_scale_config, detect_hardware

    hw = detect_hardware()
    config = auto_scale_config(config, use_recurrent=False)
"""

import copy
import logging
import os

logger = logging.getLogger(__name__)


def detect_hardware() -> dict:
    """Détecte GPU, CPU et RAM disponibles."""
    hw = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": 0.0,
        "cpu_count": os.cpu_count() or 4,
        "ram_gb": _get_ram_gb(),
    }

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw["gpu_available"] = True
            hw["gpu_name"] = props.name
            if hasattr(props, 'total_memory'):
                hw["gpu_vram_gb"] = round(props.total_memory / (1024**3), 1)
            else:
                # Fallback for older torch versions or different attributes
                hw["gpu_vram_gb"] = round(getattr(props, 'total_mem', 0) / (1024**3), 1)
    except ImportError:
        pass

    return hw


def compute_optimal_params(hw: dict, use_recurrent: bool = False) -> dict:
    """Calcule les paramètres optimaux à partir des specs hardware.

    Returns:
        dict avec n_envs, batch_size, n_steps, max_workers,
        optuna_n_jobs, optuna_n_envs_per_trial, mc_workers.
    """
    cpu_count = hw["cpu_count"]
    vram_gb = hw["gpu_vram_gb"]
    ram_gb = hw["ram_gb"]

    # --- n_envs ---
    # SubprocVecEnv: chaque env = 1 process (~400 MB RAM)
    # DummyVecEnv (RecurrentPPO): tout dans 1 process, n_envs = diversité
    usable_cores = max(cpu_count - 2, 2)

    if use_recurrent:
        n_envs = min(usable_cores, 32)
    else:
        # 256 c'est trop pour 116GB RAM. On limite à 128 max absolu.
        n_envs = min(usable_cores, 128)

    # Cap par RAM: On estime 1.2 GB par process (Data + Python overhead + Buffer)
    # Avec 116GB -> ~90 envs max
    ram_per_env = 1.2
    max_envs_by_ram = max(int((ram_gb - 6) / ram_per_env), 4)
    n_envs = min(n_envs, max_envs_by_ram)

    # --- batch_size ---
    if vram_gb >= 20:
        # 65536 est très agressif en VRAM, descendons à 32768 pour la stabilité
        batch_size = 32768
    elif vram_gb >= 10:
        batch_size = 16384
    elif vram_gb >= 6:
        batch_size = 4096
    else:
        batch_size = 1024

    # --- n_steps ---
    n_steps = 4096
    # buffer = n_envs * n_steps doit être >= batch_size
    if n_envs * n_steps < batch_size:
        n_steps = max(batch_size // n_envs, 2048)

    # --- data fetching ---
    max_workers = min(usable_cores, 8)

    # --- Optuna ---
    n_envs_per_trial = max(n_envs // 4, 2)
    optuna_n_jobs = max(usable_cores // n_envs_per_trial, 1)
    optuna_n_jobs = min(optuna_n_jobs, 4)

    # --- Monte Carlo ---
    mc_workers = usable_cores

    return {
        "n_envs": n_envs,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "max_workers": max_workers,
        "optuna_n_jobs": optuna_n_jobs,
        "optuna_n_envs_per_trial": n_envs_per_trial,
        "mc_workers": mc_workers,
    }


def auto_scale_config(config: dict, use_recurrent: bool = False) -> dict:
    """Override n_envs/batch_size/n_steps dans le config avec les valeurs optimales.

    Ne modifie pas le dict original (deep copy).
    Ne touche pas aux hyperparamètres ML (learning_rate, gamma, etc.).
    """
    config = copy.deepcopy(config)
    hw = detect_hardware()
    params = compute_optimal_params(hw, use_recurrent=use_recurrent)

    logger.info(
        f"Auto-scale: {hw['gpu_name'] or 'CPU'} "
        f"({hw['gpu_vram_gb']} GB VRAM, {hw['cpu_count']} CPUs, "
        f"{hw['ram_gb']:.0f} GB RAM)"
    )
    logger.info(
        f"  -> n_envs={params['n_envs']}, batch_size={params['batch_size']}, "
        f"n_steps={params['n_steps']}, mc_workers={params['mc_workers']}"
    )

    training = config.setdefault("training", {})
    training["n_envs"] = params["n_envs"]
    training["batch_size"] = params["batch_size"]
    training["n_steps"] = params["n_steps"]

    return config


def _get_ram_gb() -> float:
    """RAM totale en GB. Linux (/proc/meminfo) avec fallback."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (FileNotFoundError, PermissionError):
        pass

    # Windows fallback
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024**3)
    except Exception:
        pass

    return 16.0  # Fallback safe
