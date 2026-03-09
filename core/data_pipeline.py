# core/data_pipeline.py
"""Pipeline de données avec split temporel train/val/test.

Garantit qu'il n'y a aucun chevauchement temporel entre les splits,
ce qui est essentiel pour éviter le data leakage en RL financier.

Usage:
    from core.data_pipeline import DataSplitter

    data = download_data(['AAPL', 'MSFT'], period='2y', interval='1h')
    splits = DataSplitter.split(data)

    print(splits.info)
    train_data = splits.train   # Dict[str, pd.DataFrame]
    val_data   = splits.val
    test_data  = splits.test
"""

import pandas as pd
from typing import Dict, NamedTuple, Optional
from datetime import datetime


class DataSplit(NamedTuple):
    """Résultat d'un split temporel de données."""

    train: Dict[str, pd.DataFrame]
    val: Dict[str, pd.DataFrame]
    test: Dict[str, pd.DataFrame]
    info: Dict  # dates de début/fin par split


class DataSplitter:
    """Split temporel de données multi-ticker.

    Le split est fait par date (pas par shuffle) pour respecter la nature
    temporelle des séries financières. Tous les tickers sont coupés aux mêmes
    dates pivots pour garantir la cohérence.
    """

    @staticmethod
    def split(
        data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> DataSplit:
        """Split les données de chaque ticker en train/val/test.

        Args:
            data: Dict {ticker: DataFrame} avec un DatetimeIndex.
            train_ratio: Proportion pour l'entraînement (défaut 0.6).
            val_ratio: Proportion pour la validation (défaut 0.2).
            test_ratio: Proportion pour le test (défaut 0.2).

        Returns:
            DataSplit contenant .train, .val, .test et .info.

        Raises:
            ValueError: Si les ratios ne somment pas à 1.0 ou si data est vide.
        """
        if not data:
            raise ValueError("data est vide — impossible de splitter")

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Les ratios doivent sommer à 1.0, obtenu {total:.4f}"
            )

        # Trouver les dates communes à tous les tickers pour
        # couper aux mêmes indices, en utilisant le ticker le plus court.
        min_len = min(len(df) for df in data.values())
        if min_len < 10:
            raise ValueError(
                f"Pas assez de données: ticker le plus court = {min_len} bars"
            )

        train_end_idx = int(min_len * train_ratio)
        val_end_idx = int(min_len * (train_ratio + val_ratio))

        train_data: Dict[str, pd.DataFrame] = {}
        val_data: Dict[str, pd.DataFrame] = {}
        test_data: Dict[str, pd.DataFrame] = {}

        for ticker, df in data.items():
            # Tronquer au min_len pour cohérence inter-tickers
            df_trimmed = df.iloc[:min_len].copy()

            train_data[ticker] = df_trimmed.iloc[:train_end_idx].copy()
            val_data[ticker] = df_trimmed.iloc[train_end_idx:val_end_idx].copy()
            test_data[ticker] = df_trimmed.iloc[val_end_idx:].copy()

        # Info sur les splits (utilise le premier ticker comme référence)
        ref_ticker = list(data.keys())[0]
        ref_train = train_data[ref_ticker]
        ref_val = val_data[ref_ticker]
        ref_test = test_data[ref_ticker]

        info = {
            "train": {
                "start": str(ref_train.index[0]),
                "end": str(ref_train.index[-1]),
                "n_bars": len(ref_train),
                "ratio": train_ratio,
            },
            "val": {
                "start": str(ref_val.index[0]),
                "end": str(ref_val.index[-1]),
                "n_bars": len(ref_val),
                "ratio": val_ratio,
            },
            "test": {
                "start": str(ref_test.index[0]),
                "end": str(ref_test.index[-1]),
                "n_bars": len(ref_test),
                "ratio": test_ratio,
            },
            "total_bars": min_len,
            "n_tickers": len(data),
            "tickers": list(data.keys()),
        }

        return DataSplit(train=train_data, val=val_data, test=test_data, info=info)

    @staticmethod
    def validate_no_overlap(split: DataSplit) -> bool:
        """Vérifie qu'il n'y a aucun chevauchement temporel entre les splits.

        Returns:
            True si pas de chevauchement, False sinon.

        Raises:
            ValueError: Si un chevauchement est détecté.
        """
        ref_ticker = split.info["tickers"][0]

        train_end = split.train[ref_ticker].index[-1]
        val_start = split.val[ref_ticker].index[0]
        val_end = split.val[ref_ticker].index[-1]
        test_start = split.test[ref_ticker].index[0]

        if train_end >= val_start:
            raise ValueError(
                f"Chevauchement train/val: train se termine à {train_end}, "
                f"val commence à {val_start}"
            )

        if val_end >= test_start:
            raise ValueError(
                f"Chevauchement val/test: val se termine à {val_end}, "
                f"test commence à {test_start}"
            )

        return True
