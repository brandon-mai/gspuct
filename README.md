## Graph-Based Stochastic Power-UCT: Monte-Carlo Graph Search with Power Mean Estimation

[Paper's URL (arXiv)](https://arxiv.org/abs/2609.19956)

This repository contains the rough implementation and benchmark for variants of Graph-Based Stochastic Power-UCT (GSP-UCT).

---

### Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

---

### Benchmarking

Run the `benchmark-classic` command which maps to the `benchmark.py` script, or you can update the entry to map to the simple `benchmark` command (like below). The script supports **Grid Search** for comparing multiple configurations in a single run.

```bash
# Passenger Grid (Sims: 256 to 8192)
uv run benchmark --env passenger_grid --solver gbop --sims "(1024)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 70}"

uv run benchmark --env passenger_grid --solver gbop --sims "(256, 512, 1024, 2048, 4096, 8192)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 70}" --table

# Factored River Swim (Sims: 64 to 2048)
uv run benchmark --env riverswim_n4x8 --solver gbop --sims "(1024)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 35}"

uv run benchmark --env riverswim_n4x8 --solver gbop --sims "(64, 128, 256, 512, 1024, 2048)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 35}" --table

# Sysadmin Ring (Sims: 512 to 16384)
uv run benchmark --env sysadmin_n20 --solver gbop --sims "(512, 1024, 2048, 4096, 8192, 16384)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 50}" --table

# Four Rooms (Sims: 64 to 4096)
uv run benchmark --env fourrooms_n5 --solver gbop --sims "(64, 128, 256, 512, 1024, 2048, 4096)" --seeds 500 --parallel --solver_args "{'budget_strategy': 'generous', 'horizon': 50}" --table
```
---

### Arguments

- `--env`: One of five JAIR environment IDs. Supports Grid Search `"(env1, env2)"`.
- `--solver`: One of `gsp_uct`, `gsp_uct_f`, `gbop`, `ments`, or `random`.
- `--sims`: MCTS simulations per move. Supports Grid Search `"(100, 200)"`.
- `--solver_args`: Dictionary string for solver hypers. Values can be grids `"{'p': (1.0, 2.0)}"`.
- `--log`: Boolean flag. If present, saves CSVs, videos, and logs to `results/`.
- `--episodes`: Episodes per seed (default: 1).
- `--seeds`: Number of random seeds for statistical significance (default: 5).
- `--parallel`: Enable parallel episode execution.
- `--table`: Generate `result_table.txt` with a Markdown summary table.

---

### Benchmark with JAX

A limited amount of environments and algorithms were attempted to be JAX-ed for faster iteration. The current `benchmark` command is mapped to the `jax_benchmark.py` script.

```bash
# Jumanji 2048
uv run benchmark --env 2048 --solver mctx --sims 100 --seeds 1000

# Gymnax Four Rooms
uv run benchmark --env fourrooms --solver mctx --sims 100 --seeds 1000

# UCT
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard', 'p': 1.0, 'gamma': 0.95}" --table

uv run benchmark --env 2048 --solver mctx --sims "(16, 32, 64, 128)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard', 'p': 1.0, 'gamma': 0.95, 'num_chance_outcomes': 30}" --table

uv run benchmark --env frozen_lake --solver mctx --sims "(2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'standard', 'p': 1.0, 'gamma': 0.95, 'num_chance_outcomes': 3}" --table

# SPUCT
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 2.0, 'gamma': 0.95}" --table

uv run benchmark --env passenger_grid --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': 'pure_tree', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table

# GSPUCT
uv run benchmark --env 2048 --solver mctx --sims "(128)" --seeds 1000 --solver_args "{'merge_mode': 'depth_dependent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 1.2, 1.5, 2.0, 5.0, 10.0, 'inf'), 'gamma': 0.95}" --table

uv run benchmark --env passenger_grid --solver mctx --sims "(2048)" --seeds 1000 --solver_args "{'merge_mode': 'depth_dependent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table

# GSPUCTF
uv run benchmark --env fourrooms --solver mctx --sims "(128)" --seeds 1000 --solver_args "{'merge_mode': 'depth_independent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table

uv run benchmark --env frozen_lake --solver mctx --sims "(2048)" --seeds 1000 --solver_args "{'merge_mode': 'depth_independent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table

# ALL, FOUR ROOMS
uv run benchmark --env fourrooms --solver mctx --sims "(16, 32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': ('depth_dependent', 'depth_independent'), 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 2.0), 'gamma': 0.95}" --table

# ALL, 2048
uv run benchmark --env 2048 --solver mctx --sims "(32, 64, 128, 256, 512, 1024, 2048)" --seeds 1000 --solver_args "{'merge_mode': ('pure_tree', 'depth_dependent', 'depth_independent'), 'max_depth': 3, 'ucb_mode': 'spuct', 'p': (1.0, 2.0), 'gamma': 0.95, 'num_chance_outcomes': 30}" --table

# Mods ablation
uv run benchmark --env fourrooms --solver mctx --sims "(128)" --seeds 1000 --solver_args "{'merge_mode': 'depth_independent', 'max_depth': 3, 'ucb_mode': 'spuct', 'p': 1.0, 'gamma': 0.95}" --table
```

---

### Project Structure

- `core/`: MCTS logic, variants (UCT), and baselines.
- `envs/`: `GymAdapter` and Gymnasium integration.
- `utils/`: Logging and helper utilities.
- `results/`: [Internal] Output data, plots.
- `scripts/`: Entry points for benchmarking and tuning.

## License

MIT
