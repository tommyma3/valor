from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Any


def compute_returns(
    records: List[Dict[str, Any]],
    reward_field: str = "reward",
    trajectory_field: str = "trajectory_id",
    timestep_field: str = "t",
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = record.get(trajectory_field, str(idx))
        grouped[key].append(record)

    for _, traj in grouped.items():
        if timestep_field in traj[0]:
            traj.sort(key=lambda r: r[timestep_field])
        rewards = [int(r.get(reward_field, 0)) for r in traj]
        running = 0
        for r, record in zip(reversed(rewards), reversed(traj)):
            running += r
            record["return"] = running
    return records


def ensure_value_labels(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for record in records:
        if "value_label" in record:
            continue
        value = record.get("return")
        if value is None:
            value = record.get("reward", 0)
        record["value_label"] = 1 if float(value) > 0 else 0
    return records
