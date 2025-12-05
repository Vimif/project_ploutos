# tests/test_config.py
def test_config_loading():
    config = PloutosConfig.from_yaml('config/autonomous_config.yaml')
    assert config.training.device in ['cuda', 'cpu']

# tests/test_trainer.py
def test_trainer_creation():
    config = PloutosConfig()
    trainer = ModelTrainer(config)
    assert trainer.config.training.timesteps == 2_000_000
