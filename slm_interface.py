"""
NeuralPulse 2.0 - Analyst Agent Interface.

This is the public API for the Cortex Layer.
It exposes a single `AnalystAgent` class that handles the complexity
of choosing between the Mock Brain and the Real Brain.
"""

import logging
import json
import time
import sys
import os
from typing import Dict, Optional

# --- PATH FIX ---
# This ensures python can find 'reasoning_config' even if run from root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ----------------

try:
    from reasoning_config import CONFIG, SYSTEM_PROMPT
    from knowledge_base import ExpertSystem
except ImportError as e:
    # Fallback for module execution
    from .reasoning_config import CONFIG, SYSTEM_PROMPT
    from .knowledge_base import ExpertSystem

logger = logging.getLogger("AnalystAgent")


class AnalystAgent:
    def __init__(self):
        self.use_mock = CONFIG.use_mock
        self.model = None
        self.tokenizer = None

        if not self.use_mock:
            self._load_real_brain()
        else:
            logger.info("🧠 Brain Mode: MOCK (Expert System Active)")

    def _load_real_brain(self):
        """
        Attempts to load Unsloth/Transformers.
        Falls back to Mock if libraries are missing.
        """
        logger.info("🧠 Brain Mode: REAL (Loading SLM...)")
        try:
            from unsloth import FastLanguageModel
            import torch

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=CONFIG.model_name,
                max_seq_length=CONFIG.max_seq_len,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)

            self.model = model
            self.tokenizer = tokenizer
            logger.info("✅ SLM Loaded Successfully.")

        except ImportError:
            logger.error("❌ Unsloth library not found. Falling back to MOCK brain.")
            self.use_mock = True
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}. Falling back to MOCK brain.")
            self.use_mock = True

    def analyze(self, alert_vector: Dict) -> Dict:
        """
        Main entry point.
        Takes a raw VectorAlert and returns a Diagnosis JSON.
        """
        start_time = time.time()

        # 1. Mock Path
        if self.use_mock:
            result = ExpertSystem.diagnose(alert_vector)

        # 2. Real Path
        else:
            result = self._query_llm(alert_vector)

        duration = (time.time() - start_time) * 1000
        logger.info(f"🔍 Analysis Complete ({duration:.0f}ms) | Diagnosis: {result['diagnosis']}")
        return result

    def _query_llm(self, alert_vector: Dict) -> Dict:
        """
        Constructs the prompt and parses the LLM output.
        """
        # Construct natural language input from vector
        metrics = alert_vector.get("metrics", {})
        cpu = metrics.get("cpu_percent", 0)
        lat = metrics.get("latency_ms", 0)
        err = metrics.get("error_rate", 0)

        input_text = f"Alert: Anomaly Detected. Metrics: CPU={cpu:.1f}%, Latency={lat:.0f}ms, ErrorRate={err * 100:.1f}%."

        # Format with template
        prompt = f"{SYSTEM_PROMPT}\n\nInput: {input_text}\n\nOutput:"

        # Tokenize & Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=128, use_cache=True)
        decoded = self.tokenizer.batch_decode(outputs)[0]

        # Extract JSON (Simple heuristic parsing)
        try:
            # Find the first { and last }
            json_str = decoded[decoded.find("{"):decoded.rfind("}") + 1]
            return json.loads(json_str)
        except Exception:
            logger.error("Failed to parse LLM JSON. Returning fallback.")
            return ExpertSystem.diagnose(alert_vector)


# Test Harness
if __name__ == "__main__":
    agent = AnalystAgent()

    dummy_alert = {
        "metrics": {"cpu_percent": 95.5, "latency_ms": 120, "error_rate": 0.0},
        "primary_cause": "cpu_percent"
    }

    print(json.dumps(agent.analyze(dummy_alert), indent=2))