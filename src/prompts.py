import os

import yaml

from src.config import config


def _base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_system_prompt() -> str:
    path = os.path.join(_base_dir(), config.paths.prompts_dir, "system.yaml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["system_prompt"]


SYSTEM_PROMPT: str = _load_system_prompt()


if __name__ == "__main__":
    print(f"Characters: {len(SYSTEM_PROMPT)}")
    print("=" * 60)
    print(SYSTEM_PROMPT)
