"""
NeuralPulse 2.0 - SRE Reasoning Dataset Synthesizer.

This script generates synthetic training data for the Fine-Tuned SLM.
It creates pairs of (Anomaly Input) -> (Expert Diagnosis Output).

By generating this programmatically, we ensure:
1. The JSON output is always perfectly formatted.
2. We cover edge cases (e.g., High Latency but Low CPU).
3. We can generate thousands of examples instantly.

Output: reasoning_dataset.jsonl (Alpaca format for Unsloth)
"""

import json
import random
import logging
from typing import List, Dict

try:
    from .reasoning_config import DATASET_PATH, SYSTEM_PROMPT, ALPACA_TEMPLATE
except ImportError:
    from reasoning_config import DATASET_PATH, SYSTEM_PROMPT, ALPACA_TEMPLATE

logger = logging.getLogger("DatasetSynth")

# ==============================================================================
# KNOWLEDGE DOMAIN (The "Truth")
# ==============================================================================

SCENARIOS = [
    {
        "type": "CPU_SPIKE",
        "condition": lambda cpu, lat, err: cpu > 85 and lat > 200,
        "diagnosis": "Compute Resource Saturation",
        "root_cause": "Traffic surge exceeding pod capacity",
        "action": "SCALE_UP_LARGE",
        "risk": "HIGH",
        "reasoning": "CPU is critical (>85%) and causing latency cascading."
    },
    {
        "type": "MEMORY_LEAK",
        "condition": lambda cpu, lat, err: cpu < 60 and lat > 150,
        # Note: In our sim, high RAM isn't explicitly passed in text prompt usually,
        # but we can infer from "Low CPU but High Latency" often implies wait/swap.
        "diagnosis": "Memory Leak / Garbage Collection Thrashing",
        "root_cause": "Application memory leak causing swap usage",
        "action": "CLEAR_CACHE",  # Or Restart
        "risk": "MEDIUM",
        "reasoning": "Latency is high despite moderate CPU, suggesting memory pressure."
    },
    {
        "type": "DB_LOCK",
        "condition": lambda cpu, lat, err: lat > 1000 and cpu < 50,
        "diagnosis": "Database Connection Lock",
        "root_cause": "Write-heavy transaction locking queries",
        "action": "REROUTE_TRAFFIC",
        "risk": "CRITICAL",
        "reasoning": "Extreme latency with low app CPU indicates downstream DB lock."
    },
    {
        "type": "NETWORK_FAIL",
        "condition": lambda cpu, lat, err: err > 0.05,
        "diagnosis": "Upstream Network Failure",
        "root_cause": "Packet loss at ingress gateway",
        "action": "REROUTE_TRAFFIC",
        "risk": "HIGH",
        "reasoning": "High error rate (>5%) suggests connectivity loss."
    },
    {
        "type": "NORMAL_NOISE",
        "condition": lambda cpu, lat, err: cpu < 70 and lat < 150 and err < 0.01,
        "diagnosis": "System Nominal",
        "root_cause": "N/A",
        "action": "DO_NOTHING",
        "risk": "LOW",
        "reasoning": "All metrics within SLA boundaries."
    }
]


def generate_entry() -> Dict:
    """Creates a single training example."""

    # 1. Randomize Metrics
    cpu = random.uniform(10, 100)
    # Correlate latency loosely with CPU
    latency = random.uniform(20, 100) + (cpu ** 1.5) / 10
    # Add random spikes
    if random.random() < 0.1: latency += 2000

    error_rate = 0.0
    if random.random() < 0.05: error_rate = random.uniform(0.01, 0.2)

    # 2. Find matching scenario
    matched_scenario = None
    for scen in SCENARIOS:
        if scen["condition"](cpu, latency, error_rate):
            matched_scenario = scen
            break

    # Fallback
    if not matched_scenario:
        matched_scenario = SCENARIOS[-1]  # Normal

    # 3. Construct Input Text
    input_text = f"Alert: Anomaly Detected. Metrics: CPU={cpu:.1f}%, Latency={latency:.0f}ms, ErrorRate={error_rate * 100:.1f}%."

    # 4. Construct Output JSON
    output_json = {
        "diagnosis": matched_scenario["diagnosis"],
        "root_cause": matched_scenario["root_cause"],
        "action_recommendation": matched_scenario["action"],
        "risk_level": matched_scenario["risk"],
        "reasoning": matched_scenario["reasoning"],
        "confidence": round(random.uniform(0.85, 0.99), 2)
    }

    # 5. Format for Unsloth/Alpaca
    # The 'instruction' is the System Prompt
    return {
        "instruction": SYSTEM_PROMPT,
        "input": input_text,
        "output": json.dumps(output_json)
    }


def generate_dataset(num_samples: int = 500):
    """Generates and saves the JSONL file."""
    logger.info(f"🧪 Synthesizing {num_samples} reasoning examples...")

    with open(DATASET_PATH, 'w') as f:
        for _ in range(num_samples):
            entry = generate_entry()
            # We save as a list of dicts or standard jsonl.
            # Unsloth likes standard list of dicts usually, but for line-by-line reading let's do JSONL
            # Actually, let's just create a list and dump it once for standard JSON compat
            pass

    # Re-writing to save as a proper JSON list for generic loaders
    data = [generate_entry() for _ in range(num_samples)]

    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Dataset saved to {DATASET_PATH}")


if __name__ == "__main__":
    generate_dataset(500)