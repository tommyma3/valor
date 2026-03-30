#!/usr/bin/env python3
"""Check training data for issues."""

import json
from pathlib import Path

def check_data(data_path):
    data = [json.loads(line) for line in open(data_path)]

    print(f"Total examples: {len(data)}")

    issues = []
    for i, d in enumerate(data):
        # Check for empty fields
        if not d.get('action_think', '').strip():
            issues.append((i, 'empty_think'))
        if not d.get('action_memory_update', '').strip():
            issues.append((i, 'empty_memory'))
        if not d.get('action_tool_query', '').strip():
            issues.append((i, 'empty_tool'))

        # Check advantage label
        if 'advantage_label' not in d:
            issues.append((i, 'missing_advantage_label'))

        # Check extreme tokens (estimate)
        total_len = len(d.get('action_think', '')) + len(d.get('action_memory_update', '')) + len(d.get('action_tool_query', ''))
        if total_len < 50:  # Very short
            issues.append((i, 'very_short_action'))

    print(f"Issues found: {len(issues)}")
    for idx, issue in issues[:10]:  # Show first 10
        print(f"  Example {idx}: {issue}")

    # Check distribution
    adv_labels = [d['advantage_label'] for d in data if 'advantage_label' in d]
    print(f"\nAdvantage distribution: 0={adv_labels.count(0)}, 1={adv_labels.count(1)}")
    print(f"Proportion positive: {100*sum(adv_labels)/len(adv_labels):.1f}%")

    # Check returns
    returns = [d.get('return', 0) for d in data]
    print(f"\nReturn stats: min={min(returns):.2f}, max={max(returns):.2f}, mean={sum(returns)/len(returns):.2f}")

    return len(issues) == 0

if __name__ == "__main__":
    is_clean = check_data("./runs/iter_001_checkpoints/trajectories_adv.jsonl")
    print(f"\nData looks {'good' if is_clean else 'problematic'}")
