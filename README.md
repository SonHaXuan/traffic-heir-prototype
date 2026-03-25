# traffic-heir-prototype

Prototype for the paper idea:

**Encrypted Cooperative Traffic State Fusion for Privacy-Aware Signal Decision Support in Smart Cities**

This repository is a research prototype that focuses on:
- multi-intersection traffic state fusion
- privacy-aware decision support
- HE-friendly model design
- a clean path toward HEIR-based encrypted inference

## Scope of the prototype

This prototype does **not** attempt full city-scale real-time control yet. Instead it implements:
- synthetic multi-intersection traffic state generation
- local vs cooperative feature fusion
- label generation with rule-based signal decision support
- plaintext baseline model
- HE-friendly polynomial model
- export stub for HEIR-compatible inference
- evaluation utilities for local vs cooperative settings

## Initial research questions

1. Does cooperative traffic state fusion outperform local-only decision support?
2. Can a low-depth HE-friendly model preserve most of the cooperative benefit?
3. What implementation structure best supports later HEIR integration?

## Project structure

```text
traffic-heir-prototype/
├── pyproject.toml
├── README.md
├── configs/
│   └── sumo/
│       ├── two_intersections.yaml
│       ├── corridor.yaml
│       └── grid_3x3.yaml
├── src/traffic_heir/
│   ├── __init__.py
│   ├── action_space.py
│   ├── baselines.py
│   ├── config.py
│   ├── config_io.py
│   ├── synthetic.py
│   ├── labels.py
│   ├── fusion.py
│   ├── models.py
│   ├── multiclass.py
│   ├── train.py
│   ├── evaluate.py
│   ├── heir_export.py
│   ├── heir_consistency.py
│   ├── robustness.py
│   ├── sumo_data.py
│   ├── sumo_experiment.py
│   ├── sumo_scaffold.py
│   └── types.py
└── scripts/
    ├── run_prototype.py
    ├── run_with_config.py
    ├── run_sumo_experiment.py
    ├── run_action4_experiment.py
    ├── run_sumo_experiment_large_sample.sh
    ├── expand_sample_sumo.py
    ├── export_heir_stub.py
    ├── check_export_consistency.py
    ├── prepare_sumo_csv.py
    ├── action_space_demo.py
    ├── smoke_test.py
    └── sumo_scaffold.py
```

## Quick start

```bash
python3 scripts/run_prototype.py
python3 scripts/run_with_config.py configs/experiment/prototype_default.json
python3 scripts/export_heir_stub.py
python3 scripts/check_export_consistency.py
python3 scripts/smoke_test.py
python3 scripts/sumo_scaffold.py
python3 scripts/prepare_sumo_csv.py data/sumo/raw/sample_states.csv configs/sumo/sample_adjacency.json
python3 scripts/run_sumo_experiment.py data/sumo/raw/sample_states.csv configs/sumo/sample_adjacency.json
sh scripts/run_sumo_experiment_large_sample.sh
python3 scripts/action_space_demo.py
python3 scripts/run_action4_experiment.py
```

## Prototype outputs

The prototype prints:
- dataset summary
- local-only heuristic accuracy
- local MLP validation accuracy
- cooperative HE-friendly model validation accuracy
- local-vs-cooperative comparison
- a generated HEIR stub file for later encrypted inference work

## Current baseline coverage

Implemented in the current prototype:
- fixed-time baseline
- local heuristic baseline
- local max-pressure baseline
- local plaintext learned model
- cooperative plaintext learned model
- cooperative HE-friendly learned model
- ablation without interaction features
- corrected ablation without neighbor-summary features
- robustness evaluation under noisy, missing, partial-drop, and directional-corruption sensing
- initial SUMO CSV parser and cooperative-sample builder
- adjacency-aware SUMO sample construction
- HEIR export shape/consistency checks
- config-driven experiment runner
- SUMO end-to-end binary experiment runner
- expanded SUMO sample generator for non-trivial pipeline checks
- 4-action decision-support scaffold and multi-class prototype runner

## Planned next steps

1. Replace synthetic generator with SUMO-derived state extraction
2. Add robustness evaluation with missing/noisy sensors
3. Connect exported HE-friendly model to actual HEIR compilation flow
4. Add latency and communication overhead reporting
5. Expand from binary phase choice to richer signal decision support actions
