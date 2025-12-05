"""Setup Ploutos Trading System"""
from setuptools import setup, find_packages

setup(
    name="ploutos",
    version="1.0.0",
    description="Autonomous Trading System with Reinforcement Learning",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "yfinance>=0.2.28",
        "alpaca-py>=0.9.0",
        "wandb>=0.15.0",
        "psycopg2-binary>=2.9.0",
        "prometheus-client>=0.17.0",
        "pyyaml>=6.0",
        "optuna>=3.0.0",
    ],
)
