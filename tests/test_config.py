"""Tests configuration"""

import pytest
from pathlib import Path
from config.config import PloutosConfig, TrainingConfig

def test_config_default():
    """Test création config par défaut"""
    config = PloutosConfig()
    
    assert config.market.reference_ticker == "SPY"
    assert config.training.device in ['cuda', 'cpu']
    assert config.training.timesteps > 0

def test_config_from_yaml():
    """Test chargement depuis YAML"""
    yaml_path = Path('config/autonomous_config.yaml')
    
    if not yaml_path.exists():
        pytest.skip("Config YAML non trouvée")
    
    config = PloutosConfig.from_yaml(str(yaml_path))
    
    assert config.training.n_envs > 0
    assert config.validation.min_sharpe > 0

def test_config_ppo_kwargs():
    """Test export kwargs PPO"""
    config = PloutosConfig()
    kwargs = config.get_ppo_kwargs()
    
    assert 'learning_rate' in kwargs
    assert 'batch_size' in kwargs
    assert 'policy_kwargs' in kwargs
    assert 'net_arch' in kwargs['policy_kwargs']

def test_config_to_dict():
    """Test export dict"""
    config = PloutosConfig()
    d = config.to_dict()
    
    assert 'market' in d
    assert 'training' in d
    assert 'validation' in d
