"""
generate_sumo_large.py
----------------------
Generates a realistic synthetic SUMO-style CSV with:
  - 5 intersections in a grid (i0..i4)
  - 100 timesteps
  - Queue lengths driven by a simple traffic wave model
    (morning peak, afternoon lull, evening peak)
  - Phase switching logic tied to queue imbalance
  - Gaussian noise for realism

Outputs:
  data/sumo/raw/large_states.csv
  configs/sumo/large_adjacency.json
"""

import csv
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_OUT = REPO / "data" / "sumo" / "raw" / "large_states.csv"
ADJ_OUT = REPO / "configs" / "sumo" / "large_adjacency.json"

# 5-node linear-grid topology:
#  i0 -- i1 -- i2 -- i3 -- i4
INTERSECTIONS = ["i0", "i1", "i2", "i3", "i4"]
ADJACENCY = {
    "i0": ["i1"],
    "i1": ["i0", "i2"],
    "i2": ["i1", "i3"],
    "i3": ["i2", "i4"],
    "i4": ["i3"],
}

TIMESTEPS = 100
SEED = 42


def traffic_demand(t: int, intersection_idx: int) -> float:
    """
    Simulate a demand curve with morning peak (~t=20), lull (~t=50),
    evening peak (~t=75). Each intersection has a slight phase offset.
    Returns a demand multiplier in [0.3, 1.0].
    """
    offset = intersection_idx * 3  # phase offset per intersection
    t_eff = (t + offset) % TIMESTEPS
    morning = math.exp(-((t_eff - 20) ** 2) / 200)
    evening = math.exp(-((t_eff - 75) ** 2) / 150)
    base = 0.3
    return base + 0.7 * max(morning, evening)


def generate_queue(demand: float, rng: random.Random, prev: float | None = None) -> float:
    """Generate a queue length with temporal autocorrelation."""
    target = max(0.5, demand * 20 + rng.gauss(0, 1.5))
    if prev is None:
        return round(max(0.0, target), 1)
    # Smooth towards target
    val = 0.7 * prev + 0.3 * target
    return round(max(0.0, val + rng.gauss(0, 0.8)), 1)


def generate_wait(queue: float, rng: random.Random) -> float:
    """Wait time loosely proportional to queue."""
    return round(max(0.0, queue * 1.2 + rng.gauss(0, 1.0)), 1)


def main():
    rng = random.Random(SEED)

    rows = []
    # State tracking
    queues = {iid: {"q_n": 2.0, "q_s": 2.0, "q_e": 2.0, "q_w": 2.0} for iid in INTERSECTIONS}
    waits = {iid: {"w_n": 3.0, "w_s": 3.0, "w_e": 3.0, "w_w": 3.0} for iid in INTERSECTIONS}
    phases = {iid: 0 for iid in INTERSECTIONS}
    elapsed = {iid: rng.randint(0, 20) for iid in INTERSECTIONS}

    for t in range(TIMESTEPS):
        for idx, iid in enumerate(INTERSECTIONS):
            demand = traffic_demand(t, idx)

            # Update queues
            new_q = {}
            for d in ("q_n", "q_s", "q_e", "q_w"):
                new_q[d] = generate_queue(demand, rng, queues[iid][d])
            queues[iid] = new_q

            # Update waits
            new_w = {}
            for d, q in zip(("w_n", "w_s", "w_e", "w_w"), ("q_n", "q_s", "q_e", "q_w")):
                new_w[d] = generate_wait(new_q[q], rng)
            waits[iid] = new_w

            # Phase switching: switch every ~20-30 steps, biased by queue imbalance
            ns_queue = new_q["q_n"] + new_q["q_s"]
            ew_queue = new_q["q_e"] + new_q["q_w"]
            min_green = 8
            max_green = 35
            elapsed[iid] += 1
            if elapsed[iid] >= min_green:
                # Pressure-based switching
                if phases[iid] == 0 and ew_queue > ns_queue * 1.3 and elapsed[iid] >= min_green:
                    phases[iid] = 1
                    elapsed[iid] = 0
                elif phases[iid] == 1 and ns_queue > ew_queue * 1.3 and elapsed[iid] >= min_green:
                    phases[iid] = 0
                    elapsed[iid] = 0
                elif elapsed[iid] >= max_green:
                    phases[iid] = 1 - phases[iid]
                    elapsed[iid] = 0

            rows.append({
                "intersection_id": iid,
                "timestep": t,
                "q_n": new_q["q_n"],
                "q_s": new_q["q_s"],
                "q_e": new_q["q_e"],
                "q_w": new_q["q_w"],
                "w_n": new_w["w_n"],
                "w_s": new_w["w_s"],
                "w_e": new_w["w_e"],
                "w_w": new_w["w_w"],
                "phase": phases[iid],
                "elapsed": elapsed[iid],
            })

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["intersection_id", "timestep", "q_n", "q_s", "q_e", "q_w",
                  "w_n", "w_s", "w_e", "w_w", "phase", "elapsed"]
    with open(DATA_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ADJ_OUT, "w", encoding="utf-8") as f:
        json.dump(ADJACENCY, f, indent=2)

    print(f"✅ Generated {len(rows)} rows × {len(INTERSECTIONS)} intersections × {TIMESTEPS} timesteps")
    print(f"   → {DATA_OUT}")
    print(f"   → {ADJ_OUT}")

    # Quick sanity check
    phases_seen = set(r["phase"] for r in rows)
    intersections_seen = set(r["intersection_id"] for r in rows)
    print(f"   Intersections: {sorted(intersections_seen)}")
    print(f"   Phases seen: {sorted(phases_seen)}")
    q_vals = [r["q_n"] for r in rows]
    print(f"   Queue range: {min(q_vals):.1f} – {max(q_vals):.1f}")


if __name__ == "__main__":
    main()
