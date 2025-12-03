# setup.py
from setuptools import setup, find_packages

setup(
    name="ploutos",
    version="1.0.0",
    description="Système de trading multi-cerveaux IA",
    author="Votre Nom",
    author_email="votre.email@example.com",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'stable-baselines3>=2.0.0',
        'yfinance>=0.2.28',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'wandb>=0.15.0',
        'streamlit>=1.25.0',
    ],
    entry_points={
        'console_scripts': [
            'ploutos-train=scripts.train_models:main',
            'ploutos-trade=scripts.run_trader:main',
            'ploutos-backtest=scripts.backtest:main',
        ],
    },
    python_requires='>=3.8',
)
