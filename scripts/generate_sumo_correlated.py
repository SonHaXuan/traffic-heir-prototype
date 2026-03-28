"""
generate_sumo_correlated.py
---------------------------
Generates a SUMO-style CSV with genuine inter-intersection correlation via
upstream spillback.  This is the key difference from generate_sumo_large.py,
which produced each intersection's demand independently, causing neighbor
features to carry no useful signal and producing a null cooperative gain.

Design:
  - 7 intersections in a linear corridor: i0 -- i1 -- i2 -- i3 -- i4 -- i5 -- i6
  - 300 timesteps  →  2100 rows total
  - Spillback: each intersection's target queue is boosted by
      spillback_coeff × mean(upstream_neighbor_queues_prev_step)
    This creates genuine correlation: knowing i0's state helps predict what
    i1 needs to do, making cooperative neighbor features informative.
  - Shared morning/evening demand wave with small per-intersection phase offset
  - Temporal autocorrelation (0.70 * prev + 0.30 * target)
  - Gaussian measurement noise for realism

Outputs:
  data/sumo/raw/correlated_states.csv
  configs/sumo/correlated_adjacency.json
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_OUT = REPO / "data" / "sumo" / "raw" / "correlated_states.csv"
ADJ_OUT = REPO / "configs" / "sumo" / "correlated_adjacency.json"

# 7-node linear corridor topology
INTERSECTIONS = ["i0", "i1", "i2", "i3", "i4", "i5", "i6"]
ADJACENCY: dict[str, list[str]] = {
    "i0": ["i1"],
    "i1": ["i0", "i2"],
    "i2": ["i1", "i3"],
    "i3": ["i2", "i4"],
    "i4": ["i3", "i5"],
    "i5": ["i4", "i6"],
    "i6": ["i5"],
}

TIMESTEPS = 300
SEED = 42
SPILLBACK_COEFF = 0.25   # fraction of upstream mean queue that bleeds into downstream


def traffic_demand(t: int, idx: int) -> float:
    """
    Demand multiplier in [0.25, 1.0] shaped by morning peak (~t=60),
    midday lull (~t=130), evening peak (~t=220).
    Each intersection has a small phase offset to simulate wave propagation.
    """
    offset = idx * 4          # 4-timestep offset per hop down the corridor
    t_eff = (t + offset) % TIMESTEPS
    morning = math.exp(-((t_eff - 60) ** 2) / 500)
    evening = math.exp(-((t_eff - 220) ** 2) / 400)
    return 0.25 + 0.75 * max(morning, evening)


def _spillback_boost(iid: str, prev_queues: dict[str, dict[str, float]]) -> float:
    """
    Return the mean queue from upstream neighbors at the previous timestep.
    'Upstream' means the neighbors that feed traffic into this intersection —
    for a linear corridor we treat all neighbors symmetrically.
    """
    neighbors = ADJACENCY[iid]
    if not neighbors or not prev_queues:
        return 0.0
    totals = []
    for nbr in neighbors:
        if nbr in prev_queues:
            qs = prev_queues[nbr]
            totals.append((qs["q_n"] + qs["q_s"] + qs["q_e"] + qs["q_w"]) / 4.0)
    return sum(totals) / len(totals) if totals else 0.0


def _gen_queue(demand: float, spillback: float, rng: random.Random,
               prev: float | None) -> float:
    """
    Queue length = demand-driven target + upstream spillback + noise,
    smoothed with temporal autocorrelation.
    """
    target = max(0.5, demand * 18.0 + spillback * SPILLBACK_COEFF + rng.gauss(0, 1.2))
    if prev is None:
        return round(max(0.0, target), 1)
    val = 0.70 * prev + 0.30 * target
    return round(max(0.0, val + rng.gauss(0, 0.6)), 1)


def _gen_wait(queue: float, rng: random.Random) -> float:
    return round(max(0.0, queue * 1.25 + rng.gauss(0, 0.8)), 1)


def main() -> None:
    rng = random.Random(SEED)

    # Initialise state
    queues: dict[str, dict[str, float]] = {
        iid: {d: 2.0 for d in ("q_n", "q_s", "q_e", "q_w")}
        for iid in INTERSECTIONS
    }
    waits: dict[str, dict[str, float]] = {
        iid: {d: 3.0 for d in ("w_n", "w_s", "w_e", "w_w")}
        for iid in INTERSECTIONS
    }
    phases: dict[str, int] = {iid: 0 for iid in INTERSECTIONS}
    elapsed: dict[str, int] = {iid: rng.randint(0, 20) for iid in INTERSECTIONS}

    rows: list[dict] = []
    prev_queues: dict[str, dict[str, float]] = {}   # queues at t-1 for spillback

    for t in range(TIMESTEPS):
        new_queues: dict[str, dict[str, float]] = {}
        new_waits: dict[str, dict[str, float]] = {}

        for idx, iid in enumerate(INTERSECTIONS):
            demand = traffic_demand(t, idx)
            spill = _spillback_boost(iid, prev_queues)

            new_q: dict[str, float] = {}
            for d in ("q_n", "q_s", "q_e", "q_w"):
                new_q[d] = _gen_queue(demand, spill, rng, queues[iid][d])

            new_w: dict[str, float] = {}
            for wd, qd in zip(("w_n", "w_s", "w_e", "w_w"),
                              ("q_n", "q_s", "q_e", "q_w")):
                new_w[wd] = _gen_wait(new_q[qd], rng)

            new_queues[iid] = new_q
            new_waits[iid] = new_w

            # Phase switching: pressure-based with min/max green constraints
            ns_q = new_q["q_n"] + new_q["q_s"]
            ew_q = new_q["q_e"] + new_q["q_w"]
            elapsed[iid] += 1
            min_green, max_green = 8, 40
            if elapsed[iid] >= min_green:
                if phases[iid] == 0 and ew_q > ns_q * 1.25 and elapsed[iid] >= min_green:
                    phases[iid] = 1
                    elapsed[iid] = 0
                elif phases[iid] == 1 and ns_q > ew_q * 1.25 and elapsed[iid] >= min_green:
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

        queues = new_queues
        waits = new_waits
        prev_queues = {iid: dict(new_queues[iid]) for iid in INTERSECTIONS}

    # Write CSV
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["intersection_id", "timestep",
                  "q_n", "q_s", "q_e", "q_w",
                  "w_n", "w_s", "w_e", "w_w",
                  "phase", "elapsed"]
    with open(DATA_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write adjacency
    ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ADJ_OUT, "w", encoding="utf-8") as f:
        json.dump(ADJACENCY, f, indent=2)

    # Sanity checks
    q_vals = [r["q_n"] for r in rows]
    phases_seen = sorted({r["phase"] for r in rows})
    print(f"Generated {len(rows)} rows  "
          f"({len(INTERSECTIONS)} intersections × {TIMESTEPS} timesteps)")
    print(f"  Queue range: {min(q_vals):.1f} – {max(q_vals):.1f}")
    print(f"  Phases seen: {phases_seen}")
    print(f"  → {DATA_OUT}")
    print(f"  → {ADJ_OUT}")

    # Verify correlation: check that adjacent intersections have correlated NS queues
    _check_correlation(rows)


def _check_correlation(rows: list[dict]) -> None:
    """Print Pearson r between adjacent intersection NS queues as a sanity check."""
    ts_map: dict[int, dict[str, float]] = {}
    for r in rows:
        t = r["timestep"]
        if t not in ts_map:
            ts_map[t] = {}
        ts_map[t][r["intersection_id"]] = r["q_n"] + r["q_s"]

    pairs = [("i0", "i1"), ("i1", "i2"), ("i3", "i4")]
    for a, b in pairs:
        xs = [ts_map[t][a] for t in sorted(ts_map) if a in ts_map[t] and b in ts_map[t]]
        ys = [ts_map[t][b] for t in sorted(ts_map) if a in ts_map[t] and b in ts_map[t]]
        if len(xs) < 2:
            continue
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = (sum((x - mx) ** 2 for x in xs) ** 0.5) or 1e-9
        sy = (sum((y - my) ** 2 for y in ys) ** 0.5) or 1e-9
        r_val = cov / (sx * sy)
        print(f"  Pearson r({a} NS, {b} NS) = {r_val:.3f}  "
              f"(>0.3 confirms useful spillback signal)")


if __name__ == "__main__":
    main()
