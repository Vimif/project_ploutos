# training/trainer.py
"""Module d'entraînement des modèles"""
import warnings
warnings.filterwarnings('ignore')

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

from config.settings import TRAINING_CONFIG, WANDB_CONFIG, USE_GPU
from config.tickers import ALL_TICKERS
from core.environment import TradingEnv
from core.models import ModelManager
from core.utils import setup_logging, cleanup_resources, get_gpu_info, format_duration

import time
from datetime import datetime

logger = setup_logging(__name__, 'training.log')

class Trainer:
    """Entraîneur de modèles IA pour trading"""
    
    def __init__(self, config=None):
        self.config = config or TRAINING_CONFIG
        self.model_manager = ModelManager()
        
        # Info GPU
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            logger.info(f"🎮 GPU: {gpu_info['name']}")
        else:
            logger.info("⚠️  Pas de GPU, utilisation CPU")
    
    def make_env(self, ticker, rank=0):
        """Factory pour créer un environnement"""
        def _init():
            return TradingEnv(ticker)
        return _init
    
    def train_single_ticker(self, ticker: str, run_name: str = None):
        """
        Entraîner un modèle pour un ticker
        
        Args:
            ticker: Ticker à entraîner
            run_name: Nom du run WandB (optionnel)
        
        Returns:
            bool: True si succès
        """
        run_name = run_name or f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 TRAINING: {ticker}")
        logger.info(f"{'='*70}")
        
        # Init WandB
        run = wandb.init(
            project=WANDB_CONFIG['project'],
            entity=WANDB_CONFIG['entity'],
            name=run_name,
            config=self.config,
            sync_tensorboard=True,
            reinit=True
        )
        
        base_env = None
        eval_env = None
        model = None
        
        try:
            start_time = time.time()
            
            # Créer environnements
            logger.info(f"🔧 Création de {self.config['n_envs']} environnements...")
            env_fns = [self.make_env(ticker, i) for i in range(self.config['n_envs'])]
            base_env = SubprocVecEnv(env_fns, start_method='fork')
            base_env = VecMonitor(base_env)
            
            eval_env = TradingEnv(ticker)
            
            # Créer modèle
            logger.info("🤖 Création du modèle PPO...")
            device = 'cuda' if USE_GPU else 'cpu'
            policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))

            model = PPO(
                "MlpPolicy",
                base_env,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                verbose=0,
                tensorboard_log=f"./tensorboard/{ticker}",
                device=device,
                policy_kwargs=policy_kwargs
            )
            
            # Callbacks
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"./temp_models/{ticker}",
                log_path=f"./logs/{ticker}",
                eval_freq=self.config['eval_freq'],
                n_eval_episodes=self.config['n_eval_episodes'],
                deterministic=True,
                render=False
            )
            
            wandb_callback = WandbCallback(
                model_save_path=f"./temp_models/{ticker}",
                verbose=2
            )
            
            # Entraînement
            logger.info(f"🚀 Début entraînement ({self.config['total_timesteps']:,} timesteps)...")
            model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=[eval_callback, wandb_callback],
                progress_bar=True
            )
            
            # Sauvegarder
            self.model_manager.save_model(model, f"{ticker}_final")
            
            # Temps
            duration = time.time() - start_time
            logger.info(f"⏱️  Durée: {format_duration(duration)}")
            
            # GPU info
            gpu_info = get_gpu_info()
            if gpu_info['available']:
                logger.info(f"🎮 VRAM: {gpu_info['memory_allocated_gb']:.2f}GB / {gpu_info['memory_total_gb']:.2f}GB")
            
            logger.info(f"✅ {ticker} terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sur {ticker}: {e}", exc_info=True)
            return False
        
        finally:
            # Nettoyage CRITIQUE
            cleanup_resources(base_env, eval_env, model)
            
            try:
                run.finish()
            except:
                pass
            
            wandb.finish()
    
    def train_all(self, tickers=None):
        """
        Entraîner tous les tickers
        
        Args:
            tickers: Liste de tickers (ou ALL_TICKERS par défaut)
        """
        tickers = tickers or ALL_TICKERS
        
        logger.info("\n" + "="*70)
        logger.info("🚀 ENTRAÎNEMENT MULTI-TICKERS")
        logger.info("="*70)
        logger.info(f"📊 Tickers: {', '.join(tickers)}")
        logger.info(f"🔢 Timesteps par ticker: {self.config['total_timesteps']:,}")
        logger.info("="*70)
        
        start_total = time.time()
        results = {}
        
        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{idx}/{len(tickers)}] {ticker}")
            success = self.train_single_ticker(ticker)
            results[ticker] = success
        
        # Résumé
        total_duration = time.time() - start_total
        success_count = sum(results.values())
        
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ FINAL")
        logger.info("="*70)
        logger.info(f"✅ Succès: {success_count}/{len(tickers)}")
        logger.info(f"⏱️  Durée totale: {format_duration(total_duration)}")
        
        for ticker, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"   {status} {ticker}")
