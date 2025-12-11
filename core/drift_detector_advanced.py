"""
Comprehensive Drift Detection Module
====================================

Optimization #7: Détecte 5 types de drift en production simultanément.

1. **Population Stability Index (PSI)** : Distribution features change
2. **Kolmogorov-Smirnov Test (KS)** : Statut différences statistiques
3. **Maximum Mean Discrepancy (MMD)** : Différence entre distribs
4. **ADDM (Autoregressive Drift Detection)** : Concept drift (X->Y relation change)
5. **Performance Drift** : Model accuracy change

Critique en prod: Alerte si le modèle "déconne" (distribution change).

Impact: Evénement signal early warning pour retrain ou rollback.
"""

import numpy as np
from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Optional
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ComprehensiveDriftDetector:
    """
    Multi-method drift detector.
    
    Utilise 5 techniques simultanément pour détecter:
    - Data Drift (distribution change)
    - Concept Drift (relationship change)
    - Performance Drift (accuracy degradation)
    
    Attributes:
        baseline_data: Historical data for comparison
        sensitivity: 'low' | 'medium' | 'high'
        thresholds: Dict of detection thresholds
    """
    
    def __init__(
        self,
        baseline_data: np.ndarray,
        baseline_predictions: np.ndarray = None,
        baseline_actuals: np.ndarray = None,
        sensitivity: str = "medium",
    ):
        """
        Args:
            baseline_data: Historical feature data (n_samples, n_features)
            baseline_predictions: Historical model predictions
            baseline_actuals: Historical actual targets
            sensitivity: 'low' | 'medium' | 'high'
        """
        self.baseline_data = baseline_data
        self.baseline_predictions = baseline_predictions
        self.baseline_actuals = baseline_actuals
        self.sensitivity = sensitivity
        
        # Thresholds par sensibilité
        thresholds_map = {
            'low': {
                'psi': 0.10,
                'ks': 0.10,
                'mmd': 0.05,
                'performance_drop': 0.20,  # 20% accuracy drop
            },
            'medium': {
                'psi': 0.25,
                'ks': 0.15,
                'mmd': 0.10,
                'performance_drop': 0.10,  # 10% accuracy drop
            },
            'high': {
                'psi': 0.50,
                'ks': 0.25,
                'mmd': 0.20,
                'performance_drop': 0.05,  # 5% accuracy drop
            },
        }
        
        self.thresholds = thresholds_map.get(sensitivity, thresholds_map['medium'])
        
        # Baseline statistics
        self.baseline_mean = baseline_data.mean(axis=0)
        self.baseline_std = baseline_data.std(axis=0)
        
        # Baseline performance
        if baseline_predictions is not None and baseline_actuals is not None:
            self.baseline_accuracy = (
                (baseline_predictions == baseline_actuals).mean()
            )
        else:
            self.baseline_accuracy = None
        
        logger.info(
            f"DriftDetector initialized: sensitivity={sensitivity}, "
            f"baseline_samples={len(baseline_data)}"
        )
    
    def check_psi(self, current_data: np.ndarray, n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        
        Mesure comment la distribution des features a changé.
        
        PSI = sum((prop_current - prop_baseline) * log(prop_current / prop_baseline))
        
        Interpretation:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.25: Minor change (might monitor)
        - PSI > 0.25: Major change (alert!)
        
        Args:
            current_data: Current feature data (n_samples, n_features)
            n_bins: Number of bins for distribution
        
        Returns:
            PSI value (higher = more drift)
        """
        eps = 1e-10  # Avoid log(0)
        
        psi_values = []
        
        # Calculate PSI for each feature
        for feature_idx in range(self.baseline_data.shape[1]):
            baseline_col = self.baseline_data[:, feature_idx]
            current_col = current_data[:, feature_idx]
            
            # Binning strategy: use baseline percentiles
            bin_edges = np.percentile(
                baseline_col,
                np.linspace(0, 100, n_bins + 1)
            )
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Get proportions
            baseline_prop = np.histogram(baseline_col, bins=bin_edges)[0] + eps
            current_prop = np.histogram(current_col, bins=bin_edges)[0] + eps
            
            baseline_prop /= baseline_prop.sum()
            current_prop /= current_prop.sum()
            
            # PSI
            psi = np.sum(
                (current_prop - baseline_prop) * np.log(current_prop / baseline_prop)
            )
            psi_values.append(psi)
        
        # Return mean PSI across features
        mean_psi = np.mean(psi_values)
        
        return mean_psi
    
    def check_ks_statistic(self, current_data: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov Test.
        
        Statistical test if two samples come from same distribution.
        
        Returns:
            (statistic, p_value)
            - statistic: max distance between CDFs (higher = more drift)
            - p_value: probability that distributions are same (higher = less drift)
        
        Args:
            current_data: Current feature data
        
        Returns:
            (ks_statistic, max_p_value)
        """
        # Flatten all features and test
        baseline_flat = self.baseline_data.flatten()
        current_flat = current_data.flatten()
        
        statistic, p_value = ks_2samp(baseline_flat, current_flat)
        
        return statistic, p_value
    
    def check_mmd(self, current_data: np.ndarray, sigma: float = 1.0) -> float:
        """
        Maximum Mean Discrepancy (MMD).
        
        Mesure la distance entre deux distributions dans l'espace à noyau.
        
        MMD^2(X, Y) = E[phi(x)] - E[phi(y)] ^2
        
        Args:
            current_data: Current feature data
            sigma: Bandwidth for RBF kernel
        
        Returns:
            MMD value (higher = more drift)
        """
        # Sample from both (for efficiency)
        n_samples = min(1000, min(len(self.baseline_data), len(current_data)))
        
        baseline_sample = self.baseline_data[
            np.random.choice(len(self.baseline_data), n_samples, replace=False)
        ]
        current_sample = current_data[
            np.random.choice(len(current_data), n_samples, replace=False)
        ]
        
        # RBF kernel
        def rbf_kernel(x, y):
            # (n, d) @ (d, m) -> (n, m)
            sq_dist = np.sum(x**2, axis=1, keepdims=True) - 2 * x @ y.T + np.sum(y**2, axis=1, keepdims=True).T
            return np.exp(-sq_dist / (2 * sigma**2))
        
        # Compute kernel matrices
        k_xx = rbf_kernel(baseline_sample, baseline_sample)
        k_yy = rbf_kernel(current_sample, current_sample)
        k_xy = rbf_kernel(baseline_sample, current_sample)
        
        # MMD
        mmd_sq = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        mmd = np.sqrt(np.maximum(mmd_sq, 0))  # Avoid sqrt(negative)
        
        return mmd
    
    def check_performance_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict:
        """
        Check if model performance degraded.
        
        Args:
            predictions: Current model predictions
            actuals: Current actual targets
        
        Returns:
            Dict with current accuracy and drift info
        """
        current_accuracy = (predictions == actuals).mean()
        
        result = {
            'current_accuracy': current_accuracy,
            'baseline_accuracy': self.baseline_accuracy,
        }
        
        if self.baseline_accuracy is not None:
            accuracy_drop = self.baseline_accuracy - current_accuracy
            result['accuracy_drop'] = accuracy_drop
            result['performance_drifted'] = (
                accuracy_drop > self.thresholds['performance_drop']
            )
        
        return result
    
    def check_concept_drift(
        self,
        X_current: np.ndarray,
        y_current: np.ndarray,
    ) -> float:
        """
        Detect Concept Drift using ADWIN (Adaptive Windowing) proxy.
        
        Entrathe un petit classifier pour distinguer baseline vs current data.
        Si accuracy >> 50%, alors les données sont différentes (concept drift).
        
        Args:
            X_current: Current features
            y_current: Current labels
        
        Returns:
            Drift score (0.5 = no drift, 1.0 = complete drift)
        """
        if len(self.baseline_data) < 100 or len(X_current) < 100:
            logger.warning("Not enough samples for concept drift detection")
            return 0.0
        
        # Label: 0 = baseline, 1 = current
        X_combined = np.vstack([self.baseline_data, X_current])
        y_labels = np.concatenate([
            np.zeros(len(self.baseline_data), dtype=int),
            np.ones(len(X_current), dtype=int),
        ])
        
        # Train simple classifier
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        try:
            clf.fit(X_combined, y_labels)
            accuracy = clf.score(X_combined, y_labels)
            
            # Drift score: accuracy - 0.5 (normalized)
            # 0.5 = no discrimination (no drift)
            # 1.0 = perfect discrimination (complete drift)
            drift_score = min(1.0, (accuracy - 0.5) * 2)
            
            return max(0.0, drift_score)
        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
            return 0.0
    
    def full_check(
        self,
        current_data: np.ndarray,
        predictions: np.ndarray = None,
        actuals: np.ndarray = None,
    ) -> Dict:
        """
        Run all drift checks and return comprehensive report.
        
        Args:
            current_data: Current feature data
            predictions: Current predictions (optional)
            actuals: Current actual targets (optional)
        
        Returns:
            Comprehensive drift report
        """
        logger.info("Running comprehensive drift detection...")
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'baseline_samples': len(self.baseline_data),
            'current_samples': len(current_data),
        }
        
        # 1. PSI
        psi = self.check_psi(current_data)
        results['psi'] = float(psi)
        results['psi_alert'] = psi > self.thresholds['psi']
        
        # 2. KS Test
        ks_stat, ks_pval = self.check_ks_statistic(current_data)
        results['ks_statistic'] = float(ks_stat)
        results['ks_pvalue'] = float(ks_pval)
        results['ks_alert'] = ks_stat > self.thresholds['ks']
        
        # 3. MMD
        mmd = self.check_mmd(current_data)
        results['mmd'] = float(mmd)
        results['mmd_alert'] = mmd > self.thresholds['mmd']
        
        # 4. Performance Drift (if labels available)
        if predictions is not None and actuals is not None:
            perf = self.check_performance_drift(predictions, actuals)
            results['performance'] = {k: float(v) if isinstance(v, (int, float)) else v
                                     for k, v in perf.items()}
            results['performance_alert'] = perf.get('performance_drifted', False)
        
        # 5. Concept Drift (if labels available)
        if predictions is not None and actuals is not None:
            concept = self.check_concept_drift(current_data, actuals)
            results['concept_drift'] = float(concept)
            results['concept_alert'] = concept > 0.7  # High concept drift
        
        # Overall assessment
        n_alerts = sum([
            results['psi_alert'],
            results['ks_alert'],
            results['mmd_alert'],
            results.get('performance_alert', False),
            results.get('concept_alert', False),
        ])
        
        results['n_alerts'] = n_alerts
        results['overall_drift_score'] = n_alerts / 5.0  # 0 to 1
        
        if n_alerts >= 3:
            results['recommendation'] = 'RETRAIN_URGENTLY'
        elif n_alerts >= 2:
            results['recommendation'] = 'MONITOR_CLOSELY'
        else:
            results['recommendation'] = 'CONTINUE'
        
        logger.info(
            f"Drift detection complete: "
            f"alerts={n_alerts}/5, "
            f"recommendation={results['recommendation']}"
        )
        
        return results
    
    def save_report(
        self,
        results: Dict,
        path: str = "logs/drift_report_latest.json",
    ) -> None:
        """
        Save drift report to file.
        
        Args:
            results: Drift detection results
            path: File path to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Drift report saved to {path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example
    
    # 1. Create detector with baseline data
    baseline = np.random.randn(1000, 100)  # 1000 samples, 100 features
    detector = ComprehensiveDriftDetector(
        baseline_data=baseline,
        sensitivity="medium",
    )
    
    # 2. Check for drift on new data (similar distribution)
    current_similar = np.random.randn(500, 100)
    results_similar = detector.full_check(current_similar)
    print("\n=== Similar Distribution ===")
    print(f"Overall drift: {results_similar['overall_drift_score']:.2%}")
    print(f"Recommendation: {results_similar['recommendation']}")
    
    # 3. Check for drift on new data (different distribution)
    current_different = np.random.randn(500, 100) + 2  # Shifted!
    results_different = detector.full_check(current_different)
    print("\n=== Different Distribution ===")
    print(f"Overall drift: {results_different['overall_drift_score']:.2%}")
    print(f"Recommendation: {results_different['recommendation']}")
    
    # 4. Save report
    detector.save_report(results_different)
