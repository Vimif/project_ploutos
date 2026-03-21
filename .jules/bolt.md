## 2024-05-24 - [Optimize Observation Building]
**Learning:** `core/observation_builder.py` is a significant architectural bottleneck because it performs array allocations, list concatenations, and element-wise computations repeatedly at every single timestep of every trading environment.
**Action:** Use pre-allocated NumPy arrays (`np.zeros`), pre-computed slice indices in `__init__`, direct slice assignments, and apply `np.nan_to_num` and `np.clip` strictly once over the entire contiguous memory block at the end of the `build` loop to achieve significant speedups.
