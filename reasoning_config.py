"""
NeuralPulse 2.0 - Reasoning Brain Configuration.
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [REASONING_BRAIN] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ReasoningConfig")


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


PROJECT_ROOT = get_project_root()
CHECKPOINT_DIR = PROJECT_ROOT / "02_reasoning_brain" / "checkpoints"
# Path to where the Lora WOULD be if we trained it
ADAPTER_PATH = CHECKPOINT_DIR / "neuralpulse_lora"
DATASET_PATH = PROJECT_ROOT / "02_reasoning_brain" / "reasoning_dataset.jsonl"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# BRAIN MODE SELECTION (THE SAFETY SWITCH)
# ==============================================================================
# TRUE = Instant, Deterministic (Hackathon Safe Mode)
# FALSE = Tries to load LLM, falls back to Mock if failed
USE_MOCK_BRAIN = True

# Base model config (ignored if USE_MOCK_BRAIN is True)
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 1024

# ==============================================================================
# PROMPT ENGINEERING
# ==============================================================================
SYSTEM_PROMPT = """You are a Senior Site Reliability Engineer (SRE) AI.
Your task is to analyze system anomaly alerts and output a structured diagnosis in JSON format.

RULES:
1. Analyze the input metrics (CPU, Latency, Error Rate).
2. Determine the Root Cause (e.g., 'Database Saturation', 'Memory Leak').
3. Recommend a specific mitigation Action (SCALE_UP, REROUTE, CLEAR_CACHE).
4. Assign a Confidence Score (0.0 to 1.0).
5. Output MUST be valid JSON. No markdown.

Output Schema:
{
    "diagnosis": "Short description",
    "root_cause": "Specific cause",
    "action_recommendation": "SCALE_UP_SMALL|SCALE_UP_LARGE|REROUTE_TRAFFIC|CLEAR_CACHE|DO_NOTHING",
    "risk_level": "LOW|MEDIUM|HIGH",
    "reasoning": "Explanation",
    "confidence": 0.95
}
"""


@dataclass
class ReasoningConfig:
    use_mock: bool = USE_MOCK_BRAIN
    model_name: str = BASE_MODEL_NAME
    adapter_path: Path = ADAPTER_PATH
    max_seq_len: int = MAX_SEQ_LENGTH


CONFIG = ReasoningConfig()
