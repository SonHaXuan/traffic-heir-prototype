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
├── src/traffic_heir/
│   ├── __init__.py
│   ├── config.py
│   ├── synthetic.py
│   ├── labels.py
│   ├── fusion.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── heir_export.py
└── scripts/
    ├── run_prototype.py
    └── export_heir_stub.py
```

## Quick start

```bash
python3 scripts/run_prototype.py
python3 scripts/export_heir_stub.py
```

## Prototype outputs

The prototype prints:
- dataset summary
- local-only heuristic accuracy
- local MLP validation accuracy
- cooperative HE-friendly model validation accuracy
- local-vs-cooperative comparison
- a generated HEIR stub file for later encrypted inference work

## Planned next steps

1. Replace synthetic generator with SUMO-derived state extraction
2. Add topology configs: two intersections, corridor, 3x3 grid
3. Add robustness evaluation with missing/noisy sensors
4. Connect exported HE-friendly model to actual HEIR compilation flow
5. Add latency and communication overhead reporting
