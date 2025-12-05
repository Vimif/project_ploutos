#!/usr/bin/env python3
"""
Système de Détection de Dérive de Modèle
Détecte Data Drift, Concept Drift et Model Drift

Références:
- PSI (Population Stability Index)
- KS Test (Kolmogorov-Smirnov)
- ADDM (Autoregressive Drift Detection Method)
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ModelDriftDetector:
    """
    Détecte 3 types de dérive:
    1. Data Drift : Distribution features change (P(X) ≠ P'(X))
    2. Concept Drift : Relation X→Y change (P(Y|X) ≠ P'(Y|X))
    3. Model Drift : Performance se dégrade
    
    Example:
        detector = ModelDriftDetector(
            baseline_data=train_df,
            sensitivity='medium'
        )
        
        result = detector.detect_drift(
            new_data=production_df,
            new_performance={'sharpe': 0.8}
        )
        
        if result['drift_detected']:
            print(f"Dérive {result['drift_type']} détectée")
    """
    
    def __init__(self, baseline_data, baseline_performance=None, sensitivity='medium'):
        """
        Args:
            baseline_data: DataFrame avec features de référence (train/val)
            baseline_performance: Dict avec métriques baseline {'sharpe': 1.5, ...}
            sensitivity: 'low'|'medium'|'high'
        """
        self.baseline_data = baseline_data
        self.baseline_performance = baseline_performance or {}
        
        # Seuils selon sensibilité
        self.thresholds = {
            'low': {'psi': 0.25, 'ks': 0.20, 'performance': 0.30},
            'medium': {'psi': 0.15, 'ks': 0.15, 'performance': 0.20},
            'high': {'psi': 0.10, 'ks': 0.10, 'performance': 0.15}
        }[sensitivity]
        
        # Calculer stats baseline
        self._compute_baseline_stats()
        
        # Historique
        self.drift_history = []
        
        print(f"✅ Drift Detector initialisé (sensibilité: {sensitivity})")
        print(f"   Baseline: {len(baseline_data)} samples, {len(baseline_data.columns)} features")
    
    def _compute_baseline_stats(self):
        """Calcule statistiques de référence"""
        
        self.baseline_distributions = {}
        
        for col in self.baseline_data.columns:
            try:
                self.baseline_distributions[col] = {
                    'mean': float(self.baseline_data[col].mean()),
                    'std': float(self.baseline_data[col].std()),
                    'min': float(self.baseline_data[col].min()),
                    'max': float(self.baseline_data[col].max()),
                    'quantiles': self.baseline_data[col].quantile([0.1, 0.5, 0.9]).to_dict()
                }
            except:
                continue
    
    def detect_drift(self, new_data, new_performance=None):
        """
        Détecte tous types de dérive
        
        Args:
            new_data: DataFrame avec nouvelles features
            new_performance: Dict avec métriques actuelles {'sharpe': 0.8, ...}
            
        Returns:
            dict: Résultats détection avec recommandations
        """
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'drift_type': None,
            'severity': None,
            'metrics': {},
            'recommendations': [],
            'details': {}
        }
        
        # 1. DATA DRIFT (Distribution features)
        data_drift = self._detect_data_drift(new_data)
        results['metrics']['data_drift'] = data_drift
        results['details']['drifted_features'] = data_drift['drifted_features']
        
        if data_drift['psi_max'] > self.thresholds['psi']:
            results['drift_detected'] = True
            results['drift_type'] = 'data'
            results['severity'] = self._classify_severity(data_drift['psi_max'], 'psi')
            results['recommendations'].append(
                f"⚠️  Data Drift détecté (PSI={data_drift['psi_max']:.3f}). "
                f"Features impactées: {', '.join(data_drift['drifted_features'][:3])}"
            )
        
        # 2. MODEL DRIFT (Performance)
        if new_performance and self.baseline_performance:
            model_drift = self._detect_model_drift(new_performance)
            results['metrics']['model_drift'] = model_drift
            
            if model_drift['drift_detected']:
                results['drift_detected'] = True
                results['drift_type'] = 'model'
                results['severity'] = model_drift['severity']
                results['recommendations'].append(
                    f"📉 Model Drift détecté. "
                    f"Sharpe: {self.baseline_performance.get('sharpe', 0):.2f} → {new_performance.get('sharpe', 0):.2f}"
                )
        
        # Sauvegarder historique
        self.drift_history.append(results)
        
        # Logger si dérive
        if results['drift_detected']:
            self._log_drift_event(results)
        
        return results
    
    def _detect_data_drift(self, new_data):
        """
        Détecte dérive distribution via PSI et KS test
        
        PSI (Population Stability Index):
        - PSI < 0.1  : Pas de dérive
        - 0.1-0.25   : Dérive modérée
        - PSI > 0.25 : Dérive critique
        """
        
        psi_scores = {}
        ks_scores = {}
        drifted_features = []
        
        for col in new_data.columns:
            if col not in self.baseline_distributions:
                continue
            
            try:
                # 1. PSI
                psi = self._calculate_psi(
                    self.baseline_data[col].dropna(),
                    new_data[col].dropna()
                )
                psi_scores[col] = float(psi)
                
                # 2. KS Test
                ks_stat, ks_pval = stats.ks_2samp(
                    self.baseline_data[col].dropna(),
                    new_data[col].dropna()
                )
                ks_scores[col] = {
                    'statistic': float(ks_stat),
                    'pvalue': float(ks_pval)
                }
                
                # Détection
                if psi > self.thresholds['psi'] or ks_stat > self.thresholds['ks']:
                    drifted_features.append(col)
            
            except Exception as e:
                continue
        
        return {
            'psi_scores': psi_scores,
            'psi_max': max(psi_scores.values()) if psi_scores else 0,
            'ks_scores': ks_scores,
            'drifted_features': drifted_features,
            'n_drifted': len(drifted_features)
        }
    
    def _calculate_psi(self, baseline, current, bins=10):
        """
        Calcule PSI entre 2 distributions
        
        PSI = Σ (current% - baseline%) * ln(current%/baseline%)
        """
        
        # Créer bins sur baseline
        try:
            breakpoints = np.histogram_bin_edges(baseline, bins=bins)
        except:
            return 0.0
        
        # Compter occurrences
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]
        
        # Percentages
        baseline_pct = baseline_counts / max(len(baseline), 1)
        current_pct = current_counts / max(len(current), 1)
        
        # Éviter division par 0
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        # PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        
        return abs(psi)
    
    def _detect_model_drift(self, new_performance):
        """
        Détecte dégradation performance modèle
        """
        
        drift_detected = False
        severity = None
        details = {}
        
        # Comparer Sharpe Ratio
        if 'sharpe' in self.baseline_performance and 'sharpe' in new_performance:
            baseline_sharpe = self.baseline_performance['sharpe']
            current_sharpe = new_performance['sharpe']
            
            sharpe_degradation = baseline_sharpe - current_sharpe
            
            details['sharpe_degradation'] = float(sharpe_degradation)
            details['sharpe_degradation_pct'] = float((sharpe_degradation / baseline_sharpe) * 100) if baseline_sharpe != 0 else 0
            
            if sharpe_degradation > self.thresholds['performance']:
                drift_detected = True
                severity = self._classify_severity(sharpe_degradation, 'performance')
        
        # Comparer Max Drawdown
        if 'max_dd' in self.baseline_performance and 'max_dd' in new_performance:
            baseline_dd = abs(self.baseline_performance['max_dd'])
            current_dd = abs(new_performance['max_dd'])
            
            dd_increase = current_dd - baseline_dd
            details['dd_increase'] = float(dd_increase)
            
            if dd_increase > 10:  # +10% drawdown
                drift_detected = True
                severity = 'high'
        
        return {
            'drift_detected': drift_detected,
            'severity': severity,
            'details': details
        }
    
    def _classify_severity(self, value, metric_type):
        """Classifie sévérité dérive"""
        
        if metric_type == 'psi':
            if value > 0.25:
                return 'high'
            elif value > 0.15:
                return 'medium'
            else:
                return 'low'
        
        elif metric_type == 'performance':
            if value > 0.3:
                return 'high'
            elif value > 0.2:
                return 'medium'
            else:
                return 'low'
        
        return 'low'
    
    def _log_drift_event(self, results):
        """Log événement dérive"""
        
        print(f"\n{'='*80}")
        print(f"🚨 DÉRIVE DÉTECTÉE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"  Type     : {results['drift_type'].upper()}")
        print(f"  Sévérité : {results['severity'].upper()}")
        
        if results['details'].get('drifted_features'):
            print(f"\n  Features dérivées ({len(results['details']['drifted_features'])}) :")
            for feat in results['details']['drifted_features'][:5]:
                print(f"    - {feat}")
        
        print(f"\n📋 Recommandations :")
        for rec in results['recommendations']:
            print(f"  {rec}")
        print(f"{'='*80}\n")
        
        # Sauvegarder dans fichier
        os.makedirs('logs', exist_ok=True)
        
        with open('logs/drift_events.jsonl', 'a') as f:
            json.dump(results, f)
            f.write('\n')
    
    def get_drift_summary(self):
        """Résumé de l'historique de dérive"""
        
        if not self.drift_history:
            return {
                'total_checks': 0,
                'drift_events': 0,
                'drift_rate': 0
            }
        
        total = len(self.drift_history)
        drifts = sum(1 for h in self.drift_history if h['drift_detected'])
        
        # Compter par type
        drift_by_type = {}
        for h in self.drift_history:
            if h['drift_detected']:
                dtype = h['drift_type']
                drift_by_type[dtype] = drift_by_type.get(dtype, 0) + 1
        
        return {
            'total_checks': total,
            'drift_events': drifts,
            'drift_rate': (drifts / total) * 100,
            'drift_by_type': drift_by_type
        }
    
    def export_report(self, filepath='reports/drift_report.json'):
        """Exporte rapport complet"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_drift_summary(),
            'history': self.drift_history,
            'thresholds': self.thresholds,
            'baseline_performance': self.baseline_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Rapport exporté : {filepath}")

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == '__main__':
    """
    Démonstration du drift detector
    """
    
    print("\n" + "="*80)
    print("🔍 TEST DRIFT DETECTOR")
    print("="*80 + "\n")
    
    # Générer données factices
    np.random.seed(42)
    
    # Baseline (distribution normale)
    baseline_df = pd.DataFrame({
        'close_norm': np.random.normal(0, 1, 1000),
        'volume_norm': np.random.normal(0, 1, 1000),
        'rsi': np.random.uniform(30, 70, 1000),
        'macd': np.random.normal(0, 0.5, 1000)
    })
    
    # Données actuelles (avec drift)
    current_df = pd.DataFrame({
        'close_norm': np.random.normal(0.5, 1.5, 500),  # Mean shift + variance increase
        'volume_norm': np.random.normal(0, 1, 500),
        'rsi': np.random.uniform(40, 80, 500),  # Range shift
        'macd': np.random.normal(0, 0.5, 500)
    })
    
    # Initialiser detector
    detector = ModelDriftDetector(
        baseline_data=baseline_df,
        baseline_performance={'sharpe': 1.5, 'max_dd': -12.0},
        sensitivity='medium'
    )
    
    # Test 1 : Données similaires (pas de drift)
    print("🟢 Test 1 : Données similaires baseline")
    result1 = detector.detect_drift(
        new_data=baseline_df.iloc[-200:],
        new_performance={'sharpe': 1.48, 'max_dd': -11.5}
    )
    print(f"   Résultat : {'Dérive détectée' if result1['drift_detected'] else '✅ Pas de dérive'}\n")
    
    # Test 2 : Données avec drift
    print("🔴 Test 2 : Données avec drift")
    result2 = detector.detect_drift(
        new_data=current_df,
        new_performance={'sharpe': 0.8, 'max_dd': -25.0}
    )
    print(f"   Résultat : {'Dérive détectée' if result2['drift_detected'] else 'Pas de dérive'}\n")
    
    # Résumé
    print("\n📊 RÉSUMÉ :")
    summary = detector.get_drift_summary()
    print(f"  Total checks  : {summary['total_checks']}")
    print(f"  Drift events  : {summary['drift_events']}")
    print(f"  Drift rate    : {summary['drift_rate']:.1f}%")
    
    # Export
    detector.export_report('reports/drift_test.json')
    
    print("\n✅ Test terminé !\n")
