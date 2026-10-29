from pathlib import Path

from costcompose.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SAMPLE_TASKS = ROOT / "data" / "tasks" / "sample.jsonl"
SAMPLE_CORPUS = ROOT / "data" / "corpus"


def cfg(name: str):
    path = CONFIGS / f"{name}.yaml"
    if not path.exists():
        path = CONFIGS / "ablations" / f"{name}.yaml"
    c = load_config(path)
    c.caching.backend = "memory"
    return c
