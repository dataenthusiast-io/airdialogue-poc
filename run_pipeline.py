import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.pipeline import run_pipeline
from src.prompts import SYSTEM_PROMPT


def _abs(rel_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 — LLM Pipeline")
    print("=" * 60)
    print(f"\nModel: {config.model.name}  |  Temperature: {config.model.temperature}")
    print(f"Sample: {_abs(config.paths.sample)}\n")

    run_pipeline(
        sample_path=_abs(config.paths.sample),
        system_prompt=SYSTEM_PROMPT,
        output_path=_abs(config.paths.results),
    )
