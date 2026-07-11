"""
NeuralPulse 2.0 - Expert Knowledge Base (Mock Brain).

This module implements a deterministic 'Expert System' that simulates
LLM reasoning. It uses hardcoded heuristics to map Phase 1 Alerts
to Phase 3 Action contexts.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("ExpertSystem")


class ExpertSystem:
    """
    A rule-based inference engine.
    """

    @staticmethod
    def diagnose(alert_vector: Dict) -> Dict:
        """
        Input: VectorAlert dict from Phase 1.
        Output: Structured Diagnosis JSON.
        """
        metrics = alert_vector.get("metrics", {})
        # Safety check: primary_cause might be missing in some test cases
        primary_cause = alert_vector.get("primary_cause", "unknown")

        cpu = metrics.get("cpu_percent", 0)
        latency = metrics.get("latency_ms", 0)
        error = metrics.get("error_rate", 0)

        # Default response
        diagnosis = {
            "diagnosis": "Transient Noise",
            "root_cause": "Statistical jitter",
            "action_recommendation": "DO_NOTHING",
            "risk_level": "LOW",
            "reasoning": "Metrics are within acceptable variance.",
            "confidence": 0.85
        }

        # --- Heuristic Rules ---

        # Rule 1: CPU Saturation
        if primary_cause == "cpu_percent" or cpu > 90:
            diagnosis = {
                "diagnosis": "Compute Resource Exhaustion",
                "root_cause": "Traffic surge exceeding available CPU cycles",
                "action_recommendation": "SCALE_UP_LARGE",
                "risk_level": "HIGH",
                "reasoning": f"CPU is at {cpu:.1f}%, causing request queuing.",
                "confidence": 0.96
            }

        # Rule 2: Memory Leak (High RAM, Moderate CPU)
        elif primary_cause == "memory_percent" or metrics.get("memory_percent", 0) > 85:
            diagnosis = {
                "diagnosis": "Memory Leak Detected",
                "root_cause": "Application heap exhaustion / Garbage Collection death spiral",
                "action_recommendation": "CLEAR_CACHE",  # or restart
                "risk_level": "MEDIUM",
                "reasoning": "Memory usage is disproportionately high vs CPU.",
                "confidence": 0.92
            }

        # Rule 3: Database Lock (High Latency, Low CPU)
        elif primary_cause == "latency_ms" and cpu < 50:
            diagnosis = {
                "diagnosis": "Downstream Dependency Latency",
                "root_cause": "Database connection pool exhaustion or Deadlock",
                "action_recommendation": "REROUTE_TRAFFIC",
                "risk_level": "CRITICAL",
                "reasoning": f"Latency is {latency:.0f}ms while CPU is idle, indicating I/O wait.",
                "confidence": 0.98
            }

        # Rule 4: Network Failure
        elif error > 0.05:
            diagnosis = {
                "diagnosis": "Service Availability Degradation",
                "root_cause": "Network packet loss or gateway failure",
                "action_recommendation": "REROUTE_TRAFFIC",
                "risk_level": "HIGH",
                "reasoning": f"Error rate spiked to {error * 100:.1f}%.",
                "confidence": 0.99
            }

        return diagnosis