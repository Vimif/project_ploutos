#!/usr/bin/env python3
"""
Verify V6 Advanced Installation
===============================

Comprehensive checks to ensure all V6 patches are correctly installed.

Usage:
    python scripts/verify_v6_installation.py
"""

import sys
import os
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class V6Verifier:
    """Vérifie l'installation de V6"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def print_header(self, title):
        """Print section header"""
        print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
        print(f"{BOLD}{BLUE}  {title}{RESET}")
        print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    def check(self, name, condition, details=""):
        """Check a condition and print result"""
        if condition:
            print(f"{GREEN}✅{RESET} {name}")
            if details:
                print(f"   {details}")
            self.checks_passed += 1
        else:
            print(f"{RED}❌{RESET} {name}")
            if details:
                print(f"   {RED}{details}{RESET}")
            self.checks_failed += 1
    
    def warning(self, msg):
        """Add a warning"""
        print(f"{YELLOW}⚠️ {RESET} {msg}")
        self.warnings.append(msg)
    
    def verify_files_exist(self):
        """Vérify que tous les fichiers existent"""
        self.print_header("1️⃣ FILE STRUCTURE CHECK")
        
        files_to_check = [
            ("core/observation_builder_v7.py", "3D Observation Builder"),
            ("core/reward_calculator_advanced.py", "Differential Sharpe Reward"),
            ("core/normalization.py", "Adaptive Normalizer"),
            ("core/transformer_encoder.py", "Transformer Feature Extractor"),
            ("core/replay_buffer_prioritized.py", "Prioritized Replay Buffer"),
            ("core/drift_detector_advanced.py", "Drift Detector"),
            ("core/ensemble_trader.py", "Ensemble Trader"),
            ("config/training_v6_extended_optimized.yaml", "V6 Configuration"),
            ("scripts/train_v6_extended_with_optimizations.py", "V6 Training Script"),
            ("scripts/feature_importance_analysis.py", "Feature Importance Script"),
            ("scripts/walk_forward_validator.py", "Walk-Forward Validator"),
            ("scripts/apply_v6_patches.py", "Automatic Patch Script"),
            ("Makefile", "Convenience Commands"),
        ]
        
        for filepath, description in files_to_check:
            exists = os.path.exists(filepath)
            self.check(
                f"{description}",
                exists,
                f"Path: {filepath}" if not exists else f"Found at {filepath}"
            )
    
    def verify_imports(self):
        """Vérify que tous les modules peuvent être importés"""
        self.print_header("2️⃣ IMPORTS CHECK")
        
        modules = [
            ("core.observation_builder_v7", "ObservationBuilderV7"),
            ("core.reward_calculator_advanced", "DifferentialSharpeRewardCalculator"),
            ("core.normalization", "AdaptiveNormalizer"),
            ("core.transformer_encoder", "TransformerFeatureExtractor"),
            ("core.replay_buffer_prioritized", "PrioritizedReplayBuffer"),
            ("core.drift_detector_advanced", "ComprehensiveDriftDetector"),
            ("core.ensemble_trader", "EnsembleTrader"),
        ]
        
        for module_name, class_name in modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.check(
                    f"Import {class_name}",
                    True,
                    f"Successfully imported from {module_name}"
                )
            except Exception as e:
                self.check(
                    f"Import {class_name}",
                    False,
                    f"Error: {str(e)[:60]}"
                )
    
    def verify_environment_patches(self):
        """Vérify que l'environnement a été patché"""
        self.print_header("3️⃣ ENVIRONMENT PATCHES CHECK")
        
        env_file = "core/universal_environment_v6_better_timing.py"
        
        if not os.path.exists(env_file):
            self.check("Environment file exists", False, f"{env_file} not found")
            return
        
        with open(env_file, 'r') as f:
            content = f.read()
        
        patches = [
            ("from core.observation_builder_v7 import", "ObservationBuilderV7 import"),
            ("from core.reward_calculator_advanced import", "DifferentialSharpeRewardCalculator import"),
            ("from core.normalization import", "AdaptiveNormalizer import"),
            ("self.reward_calc = DifferentialSharpeRewardCalculator", "Reward calculator initialization"),
            ("self.obs_builder = ObservationBuilderV7", "Observation builder initialization"),
            ("self.normalizer = AdaptiveNormalizer", "Normalizer initialization"),
            ("def _get_observation(self) -> np.ndarray:", "_get_observation method exists"),
            ("self.obs_builder.build_observation", "_get_observation delegates to obs_builder"),
            ("self.reward_calc.calculate", "Reward calculation uses new calculator"),
        ]
        
        for code_snippet, description in patches:
            found = code_snippet in content
            self.check(
                description,
                found,
                f"Looking for: {code_snippet[:40]}..."
            )
    
    def verify_config(self):
        """Vérify la configuration V6"""
        self.print_header("4️⃣ CONFIGURATION CHECK")
        
        config_file = "config/training_v6_extended_optimized.yaml"
        
        if not os.path.exists(config_file):
            self.check("Config file exists", False, f"{config_file} not found")
            return
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.check(
                "YAML parsing",
                config is not None,
                f"Successfully loaded {config_file}"
            )
            
            # Check key sections
            sections = ['training', 'environment', 'optimizations', 'monitoring']
            for section in sections:
                self.check(
                    f"Config section '{section}'",
                    section in config,
                    f"Section {'found' if section in config else 'missing'}"
                )
            
            # Check important settings
            if 'training' in config:
                timesteps = config['training'].get('timesteps', 0)
                self.check(
                    "Training timesteps configured",
                    timesteps > 0,
                    f"Timesteps: {timesteps:,}"
                )
            
            if 'optimizations' in config:
                opt_count = len(config['optimizations'])
                self.check(
                    "Optimizations configured",
                    opt_count >= 7,
                    f"Number of optimizations: {opt_count}"
                )
        
        except Exception as e:
            self.check("Config parsing", False, f"Error: {str(e)[:60]}")
    
    def verify_dependencies(self):
        """Vérify les dépendances"""
        self.print_header("5️⃣ DEPENDENCIES CHECK")
        
        dependencies = [
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("torch", "PyTorch"),
            ("gymnasium", "Gymnasium"),
            ("stable_baselines3", "Stable-Baselines3"),
            ("yaml", "PyYAML"),
            ("sklearn", "Scikit-Learn"),
            ("scipy", "SciPy"),
        ]
        
        for module_name, display_name in dependencies:
            try:
                __import__(module_name)
                self.check(f"{display_name}", True)
            except ImportError:
                self.check(f"{display_name}", False, f"Module '{module_name}' not installed")
    
    def verify_hardware(self):
        """Vérify le hardware"""
        self.print_header("6️⃣ HARDWARE CHECK")
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            self.check(
                "CUDA available",
                cuda_available,
                f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'Not found'}"
            )
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                self.check(
                    "GPU count",
                    device_count > 0,
                    f"Found {device_count} GPU(s)"
                )
                
                # Check memory
                try:
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"   VRAM Allocated: {mem_allocated:.2f} GB")
                    print(f"   VRAM Reserved: {mem_reserved:.2f} GB")
                except:
                    pass
            else:
                self.warning("No CUDA GPU found - Training will be very slow")
        
        except Exception as e:
            self.check("Hardware detection", False, str(e))
    
    def run_all_checks(self):
        """Execute all verification checks"""
        print(f"{BOLD}{BLUE}")
        print("")
        print("  🚀 PLOUTOS V6 ADVANCED - INSTALLATION VERIFICATION")
        print("")
        print(f"{RESET}")
        
        self.verify_files_exist()
        self.verify_imports()
        self.verify_environment_patches()
        self.verify_config()
        self.verify_dependencies()
        self.verify_hardware()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of checks"""
        self.print_header("📋 VERIFICATION SUMMARY")
        
        total = self.checks_passed + self.checks_failed
        percentage = (self.checks_passed / total * 100) if total > 0 else 0
        
        print(f"{BOLD}Checks Passed: {GREEN}{self.checks_passed}{RESET}")
        print(f"{BOLD}Checks Failed: {RED}{self.checks_failed}{RESET}")
        print(f"{BOLD}Success Rate: {GREEN}{percentage:.0f}%{RESET}")
        
        if self.warnings:
            print(f"\n{BOLD}Warnings:{RESET}")
            for warning in self.warnings:
                print(f"  {YELLOW}⚠️ {RESET}{warning}")
        
        print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
        
        if self.checks_failed == 0:
            print(f"\n{GREEN}{BOLD}✅ ALL CHECKS PASSED! V6 is ready to use.{RESET}")
            print(f"\n{BOLD}Next steps:{RESET}")
            print(f"  1. Run: {BLUE}make test-patches{RESET}")
            print(f"  2. Run: {BLUE}make train-v6-test{RESET} (quick 5M test)")
            print(f"  3. Run: {BLUE}make train-v6-full{RESET} (full 50M training)")
            return 0
        else:
            print(f"\n{RED}{BOLD}❌ SOME CHECKS FAILED! Please fix the issues above.{RESET}")
            print(f"\n{BOLD}Troubleshooting:{RESET}")
            print(f"  - Check the error messages above")
            print(f"  - Run: {BLUE}python scripts/apply_v6_patches.py{RESET} to re-apply patches")
            print(f"  - Check: {BLUE}git diff core/universal_environment_v6_better_timing.py{RESET}")
            return 1


if __name__ == "__main__":
    verifier = V6Verifier()
    exit_code = verifier.run_all_checks()
    sys.exit(exit_code)
