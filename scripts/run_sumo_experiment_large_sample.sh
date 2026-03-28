#!/bin/sh
set -eu
python3 scripts/expand_sample_sumo.py data/sumo/raw/sample_states.csv data/sumo/processed/sample_states_expanded.csv
python3 scripts/run_sumo_experiment.py data/sumo/processed/sample_states_expanded.csv configs/sumo/sample_adjacency.json
