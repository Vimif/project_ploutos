# Ploutos Repo Context

## Quick Commands

- Install dependencies: `pip install -e ".[dev,training,web]"`
- Run all tests: `pytest`
- Run a focused env test file: `pytest tests/test_trading_env_v8.py`
- Check formatting: `black --check .`
- Run lint: `ruff check .`
- Run the full pipeline: `python scripts/run_pipeline.py --config config/config.yaml --auto-scale --ensemble 3`
- Train V9 PPO: `python training/train.py --config config/config.yaml --auto-scale`
- Train recurrent PPO: `python training/train.py --config config/config.yaml --recurrent --auto-scale`
- Train an ensemble: `python training/train.py --config config/config.yaml --ensemble 3 --auto-scale`
- Run hyperparameter search: `python scripts/optimize_hyperparams.py --config config/config.yaml --n-trials 50 --auto-scale`
- Run robustness tests: `python scripts/robustness_tests.py --model models/<fold>/model.zip --all --auto-scale`
- Start paper trading: `python scripts/paper_trade.py`

## High-Signal File Map

- `config/`
  - `config.yaml`: runtime and training configuration
  - `hardware.py`: auto-scale overrides
  - `schema.py`: YAML validation and cross-field checks
  - `settings.py`: paths and broker settings
- `core/`
  - `environment.py`: V9 trading environment
  - `env_config.py`: structured environment config
  - `observation_builder.py`: observation vector construction
  - `reward_calculator.py`: DSR reward logic
  - `features.py`: technical indicators and feature generation
  - `data_pipeline.py`: split logic and feature pipeline
  - `ensemble.py`: ensemble inference and confidence filtering
  - `transaction_costs.py`: slippage, spread, and commission model
  - `shared_memory_manager.py`: shared-memory path for vectorized envs
- `training/train.py`: walk-forward training entry point
- `scripts/`: full pipeline, optimization, robustness, and paper trading CLIs
- `trading/`: broker integrations
- `tests/`: pytest coverage across environment, features, config, ensemble, and pipeline behavior

## Repo Guardrails

- Use YAML config loading patterns already in the repo; V9 relies on `yaml.safe_load()` in training scripts.
- Prefer `--auto-scale`; treat the cloud config as a manual override path, not the default.
- Keep feature computation inside each walk-forward fold to avoid look-ahead bias.
- Use `pd.api.types.is_numeric_dtype()` instead of explicit dtype tuples when selecting numeric feature columns.
- Keep the shared-memory path and raw-data path behaviorally aligned.
- Preserve the Polars index round-trip through `__date_idx` when touching feature engineering internals.
- Remember that recurrent PPO uses `DummyVecEnv`, while PPO uses `SubprocVecEnv`.
- Treat `models/` as gitignored local output and `data_cache/` as disposable cached market data.
- Expect WandB to be off unless `wandb.enabled: true` is set in YAML.

## Testing Notes

- Prefer the smallest relevant test target first, especially for environment or config changes.
- Expect some historical file names to remain, such as `tests/test_trading_env_v8.py`, even though the repo is on V9.
- Run broader checks only when the change surface justifies them.

## Related Docs

- `AGENTS.md`: repo commands, architecture, and gotchas
- `docs/coding_guidelines.md`: local engineering principles
- `docs/ROADMAP.md`: roadmap and planned direction
