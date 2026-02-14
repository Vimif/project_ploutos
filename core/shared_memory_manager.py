import logging

import numpy as np
import pandas as pd
from multiprocessing.shared_memory import SharedMemory
import pickle
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)

class SharedDataManager:
    """
    Gère le stockage de datasets volumineux en mémoire partagée (SharedMemory).
    Permet à N processus de lire les mêmes données sans duplication RAM (Zero-Copy).
    """

    def __init__(self):
        self._shm_registry: List[SharedMemory] = []

    def put_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Stocke un dictionnaire de DataFrames en mémoire partagée.
        
        Args:
            data_dict: Dict {ticker: DataFrame}
            
        Returns:
            metadata: Dict contenant les infos nécessaires pour reconstruire les arrays
                      (nom shm, shape, dtype, colonnes, index).
        """
        metadata = {}
        
        for key, df in data_dict.items():
            # 1. Convertir DataFrame en Numpy (float32 pour économiser RAM)
            # On ne garde que les colonnes numériques pour la shm pure
            # Les colonnes et index sont stockés en metadata
            
            # Assurer que tout est float32
            numeric_df = df.select_dtypes(include=[np.number]).astype(np.float32)
            arr = numeric_df.values
            
            # 2. Créer SharedMemory
            logger.debug(f"Creating shm size: {arr.nbytes}")
            shm = SharedMemory(create=True, size=arr.nbytes)
            logger.debug(f"Created shm: {shm.name}")
            self._shm_registry.append(shm)
            
            # 3. Copier les données
            shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shared_arr[:] = arr[:]  # Copy data
            logger.debug("Copied data to shm")
            
            # 4. Stocker metadata
            metadata[key] = {
                "shm_name": shm.name,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "columns": numeric_df.columns.tolist(),
                "index": df.index.tolist(), # Attention: index peut être gros, mais c'est du pickle
            }
            
        return metadata

    def cleanup(self):
        """Libère la mémoire partagée (à appeler par le processus principal à la fin)."""
        for shm in self._shm_registry:
            try:
                shm.close()
                shm.unlink() # Détruit le bloc mémoire
            except Exception as e:
                logger.warning(f"Cleanup shm {shm.name}: {e}")
        self._shm_registry.clear()

def load_shared_array(metadata_item: Dict[str, Any]) -> Tuple[SharedMemory, np.ndarray]:
    """
    Charge un array numpy depuis la SHM en mode Zero-Copy.
    
    Returns:
        (shm, arr): On retourne l'objet shm pour qu'il ne soit pas GC (et fermé).
                    L'appelant doit conserver 'shm' tant qu'il utilise 'arr'.
    """
    shm = SharedMemory(name=metadata_item["shm_name"])
    arr = np.ndarray(
        metadata_item["shape"], 
        dtype=metadata_item["dtype"], 
        buffer=shm.buf
    )
    return shm, arr

def load_shared_data(metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Reconstruit les DataFrames (avec copie). legacy V8 compatible."""
    reconstructed_data = {}
    
    for key, meta in metadata.items():
        try:
            logger.debug(f"Loading shm: {meta['shm_name']}")
            # 1. Connecter à la Shm existante
            shm = SharedMemory(name=meta["shm_name"])
            
            # 2. Créer array numpy view
            arr = np.ndarray(
                meta["shape"], 
                dtype=meta["dtype"], 
                buffer=shm.buf
            )
            
            # 3. Reconstruire DataFrame (Pandas va copier ici, mais c'est léger car 1 env à la fois)
            # Pour une vraie optimisation Zero-Copy, l'env devrait utiliser l'array numpy direct.
            # Mais pour la compatibilité V8, on refait un DF.
            # L'avantage : le gros blob reste en Shm, on ne duplique que le petit batch actif localement.
            
            # NOTE: Pour V9 pur, l'env devrait consommer 'arr' directement.
            # Ici on fait une transition douce. 
            df = pd.DataFrame(
                arr, # Ceci crée une COPIE si on ne fait pas attention, mais pandas est tricky.
                index=meta["index"],
                columns=meta["columns"],
                copy=True # Force la copie pour détacher de la SHM
            )
            logger.debug(f"Loaded df: {list(df.columns)}")
            
            # Note: shm ne doit pas être close() ici si on veut garder l'accès buffer,
            # mais pandas copie souvent les données à l'init.
            # Pour V9 optimisé, on gardera shm ouverte.
            shm.close()
            
            reconstructed_data[key] = df
            
        except Exception as e:
            logger.error(f"Error loading shared data for {key}: {e}")
            
    return reconstructed_data
