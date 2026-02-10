"""
Configuration centralisée
Utilise dataclasses pour typage fort
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import json

@dataclass
class MarketConfig:
    """Configuration détection de marché"""
    reference_ticker: str = "SPY"
    lookback_days: int = 90

@dataclass
class AssetsConfig:
    """Configuration sélection d'assets"""
    n_assets: int = 10
    min_volume: int = 1_000_000

@dataclass
class OptimizationConfig:
    """Configuration optimisation hyperparamètres"""
    n_trials: int = 50
    n_jobs: int = 1
    timeout: Optional[int] = None

@dataclass
class TrainingConfig:
    """Configuration entraînement"""
    timesteps: int = 2_000_000
    n_envs: int = 8
    device: str = "cuda"
    
    # Hyperparamètres PPO
    learning_rate: float = 1e-4
    n_steps: int = 8192
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Architecture réseau
    net_arch_pi: List[int] = field(default_factory=lambda: [512, 512])
    net_arch_vf: List[int] = field(default_factory=lambda: [512, 512])

@dataclass
class ValidationConfig:
    """Configuration validation"""
    min_sharpe: float = 1.5
    test_days: int = 90
    max_steps: int = 2000

@dataclass
class MarketScanConfig:
    """Configuration scan de marché"""
    enabled: bool = False
    min_market_cap: float = 1e9
    min_volume: float = 1e6
    max_stocks: int = 500
    cache_results: bool = True

@dataclass
class SectorScanConfig:
    """Configuration scan S&P 500 par secteur GICS"""
    enabled: bool = True
    stocks_per_sector: int = 2
    lookback_days: int = 252
    risk_free_rate: float = 0.04
    cache_max_age_days: int = 30
    scan_results_path: str = "data/sp500_cache/latest_scan.json"
    parallel_workers: int = 5
    min_data_days: int = 100

@dataclass
class PloutosConfig:
    """Configuration complète du système Ploutos"""

    market: MarketConfig = field(default_factory=MarketConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    market_scan: MarketScanConfig = field(default_factory=MarketScanConfig)
    sector_scan: SectorScanConfig = field(default_factory=SectorScanConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PloutosConfig':
        """
        Charge configuration depuis YAML
        
        Args:
            path: Chemin vers fichier YAML
            
        Returns:
            PloutosConfig configuré
        """
        yaml_path = Path(path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config non trouvée: {path}")
        
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        return cls(
            market=MarketConfig(**data.get('market', {})),
            assets=AssetsConfig(**data.get('assets', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            training=TrainingConfig(**data.get('training', {})),
            validation=ValidationConfig(**data.get('validation', {})),
            market_scan=MarketScanConfig(**data.get('market_scan', {})),
            sector_scan=SectorScanConfig(**data.get('sector_scan', {})),
        )

    @classmethod
    def from_json(cls, path: str) -> 'PloutosConfig':
        """Charge depuis JSON"""
        with open(path) as f:
            data = json.load(f)

        return cls(
            market=MarketConfig(**data.get('market', {})),
            assets=AssetsConfig(**data.get('assets', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            training=TrainingConfig(**data.get('training', {})),
            validation=ValidationConfig(**data.get('validation', {})),
            market_scan=MarketScanConfig(**data.get('market_scan', {})),
            sector_scan=SectorScanConfig(**data.get('sector_scan', {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export en dictionnaire"""
        return {
            'market': asdict(self.market),
            'assets': asdict(self.assets),
            'optimization': asdict(self.optimization),
            'training': asdict(self.training),
            'validation': asdict(self.validation),
            'market_scan': asdict(self.market_scan),
            'sector_scan': asdict(self.sector_scan),
        }
    
    def save_yaml(self, path: str):
        """Sauvegarde en YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        """Sauvegarde en JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_ppo_kwargs(self) -> Dict[str, Any]:
        """
        Retourne kwargs pour PPO
        
        Returns:
            Dict prêt pour PPO(**kwargs)
        """
        return {
            'learning_rate': self.training.learning_rate,
            'n_steps': self.training.n_steps,
            'batch_size': self.training.batch_size,
            'n_epochs': self.training.n_epochs,
            'gamma': self.training.gamma,
            'gae_lambda': self.training.gae_lambda,
            'clip_range': self.training.clip_range,
            'ent_coef': self.training.ent_coef,
            'vf_coef': self.training.vf_coef,
            'max_grad_norm': self.training.max_grad_norm,
            'policy_kwargs': {
                'net_arch': {
                    'pi': self.training.net_arch_pi,
                    'vf': self.training.net_arch_vf
                }
            }
        }
