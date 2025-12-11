#!/usr/bin/env python3
"""
Feature Importance Analysis Script
==================================

Optimization #4: Analyse quelles features l'IA utilise vraiment.

Méthodes:
1. **Permutation Importance**: Shuffle chaque feature et mesurer la chute de performance
2. **SHAP Values** (optionnel): Plus coûteux mais plus précis
3. **Gradient-based**: Utile pour les réseaux de neurones

Output: rapport JSON + graphiques pour identifier les features inutiles.

Usage:
    python scripts/feature_importance_analysis.py \
        --model models/v6_extended/stage_3_final.zip \
        --env training_env \
        --n-samples 1000 \
        --output results/feature_importance.json
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import argparse
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

from stable_baselines3 import PPO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureImportanceAnalyzer:
    """
    Analyze which features are actually used by the model.
    """
    
    def __init__(self, model_path: str, n_features: int = 1293):
        """
        Args:
            model_path: Path to trained PPO model
            n_features: Number of input features
        """
        logger.info(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        self.n_features = n_features
        self.baseline_reward = None
        
        logger.info(✅ Model loaded")
    
    def evaluate_episode(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> float:
        """
        Evaluate model on environment.
        
        Args:
            env: Gymnasium environment
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
        
        Returns:
            Mean episode reward
        """
        total_reward = 0.0
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def permutation_importance(
        self,
        env,
        n_samples: int = 1000,
        n_eval_episodes: int = 5,
    ) -> Dict[str, float]:
        """
        Calculate permutation importance for each feature.
        
        Idea: Shuffle each feature and measure performance drop.
        Features that hurt performance when shuffled = important.
        
        Args:
            env: Training environment
            n_samples: Number of samples to test
            n_eval_episodes: Episodes to evaluate per feature
        
        Returns:
            Dict of {feature_idx: importance_score}
        """
        logger.info("\nCalculating baseline performance...")
        self.baseline_reward = self.evaluate_episode(
            env,
            n_episodes=n_eval_episodes,
        )
        logger.info(f"Baseline reward: {self.baseline_reward:.4f}")
        
        importances = np.zeros(self.n_features)
        
        logger.info(f"\nPermutation importance (shuffling each feature)...")
        
        # Test each feature
        for feature_idx in tqdm(range(self.n_features), desc="Features"):
            corrupted_rewards = []
            
            for _ in range(n_eval_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    # Shuffle feature
                    obs_corrupted = obs.copy()
                    if len(obs_corrupted.shape) == 1:
                        obs_corrupted[feature_idx] = np.random.randn()
                    else:
                        obs_corrupted[:, feature_idx] = np.random.randn(
                            obs_corrupted.shape[0]
                        )
                    
                    action, _ = self.model.predict(
                        obs_corrupted,
                        deterministic=True,
                    )
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                corrupted_rewards.append(episode_reward)
            
            # Importance = decrease in performance
            mean_corrupted = np.mean(corrupted_rewards)
            importance = self.baseline_reward - mean_corrupted
            importances[feature_idx] = max(0, importance)  # Non-negative
        
        return {idx: float(imp) for idx, imp in enumerate(importances)}
    
    def analyze(self, env, n_samples: int = 1000) -> Dict:
        """
        Run full feature importance analysis.
        
        Args:
            env: Environment
            n_samples: Samples to evaluate
        
        Returns:
            Analysis report
        """
        importances = self.permutation_importance(env, n_samples=n_samples)
        
        # Sort
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Classify features
        importance_values = np.array(list(importances.values()))
        threshold_high = np.percentile(importance_values, 75)
        threshold_medium = np.percentile(importance_values, 50)
        threshold_low = np.percentile(importance_values, 25)
        
        high_importance = [i for i, imp in importances.items() if imp >= threshold_high]
        medium_importance = [
            i for i, imp in importances.items()
            if threshold_medium <= imp < threshold_high
        ]
        low_importance = [
            i for i, imp in importances.items()
            if threshold_low <= imp < threshold_medium
        ]
        unused = [i for i, imp in importances.items() if imp < threshold_low]
        
        report = {
            'baseline_reward': float(self.baseline_reward),
            'n_features_total': self.n_features,
            'feature_importances': importances,
            'top_10_features': sorted_features[:10],
            'bottom_10_features': sorted_features[-10:],
            'statistics': {
                'mean_importance': float(importance_values.mean()),
                'std_importance': float(importance_values.std()),
                'max_importance': float(importance_values.max()),
                'min_importance': float(importance_values.min()),
            },
            'classification': {
                'high_importance_count': len(high_importance),
                'medium_importance_count': len(medium_importance),
                'low_importance_count': len(low_importance),
                'unused_count': len(unused),
                'high_importance_features': high_importance,
                'unused_features': unused,
            },
            'recommendations': {
                'can_remove': f"Features {unused} are unused (importance < {threshold_low:.4f})",
                'monitor': f"Features {low_importance} have low importance",
                'critical': f"Top 10 features account for {sum(imp for _, imp in sorted_features[:10]) / sum(importance_values.values()):.1%} of total importance",
            },
        }
        
        return report
    
    def plot_importance(
        self,
        importances: Dict[int, float],
        top_n: int = 50,
        output_path: str = "feature_importance.png",
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importances: Feature importance dict
            top_n: Number of top features to plot
            output_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available, skipping plots")
            return
        
        # Sort and select top
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        
        features = [f"Feature {idx}" for idx, _ in sorted_features]
        values = [imp for _, imp in sorted_features]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(features, values)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Plot saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance of trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dummy",
        help="Environment to use for evaluation",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/feature_importance.json",
        help="Output file path",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="results/feature_importance.png",
        help="Plot output path",
    )
    
    args = parser.parse_args()
    
    # Create dummy environment for testing
    # In practice, you'd use your actual trading environment
    try:
        from core.universal_environment_v6_better_timing import TradingEnvironmentV6
        env = TradingEnvironmentV6(assets=["SPY"])
    except Exception as e:
        logger.error(f"Could not create environment: {e}")
        logger.info("Using dummy CartPole environment for demo...")
        import gymnasium as gym
        env = gym.make("CartPole-v1")
    
    # Analyze
    analyzer = FeatureImportanceAnalyzer(args.model, n_features=1293)
    report = analyzer.analyze(env, n_samples=args.n_samples)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total features: {report['n_features_total']}")
    print(f"Baseline reward: {report['baseline_reward']:.4f}")
    print(f"\nFeature Classification:")
    print(f"  High importance: {report['classification']['high_importance_count']}")
    print(f"  Medium importance: {report['classification']['medium_importance_count']}")
    print(f"  Low importance: {report['classification']['low_importance_count']}")
    print(f"  Unused: {report['classification']['unused_count']}")
    print(f"\nTop 10 Features:")
    for idx, (feat_idx, imp) in enumerate(report['top_10_features'], 1):
        print(f"  {idx}. Feature {feat_idx}: {imp:.4f}")
    print(f"\nUnused Features (can remove): {report['classification']['unused_features'][:10]}...")
    print(f"\nRecommendations:")
    for key, rec in report['recommendations'].items():
        print(f"  - {rec}")
    print("="*70)
    
    # Plot
    analyzer.plot_importance(
        report['feature_importances'],
        top_n=50,
        output_path=args.plot,
    )


if __name__ == "__main__":
    main()
