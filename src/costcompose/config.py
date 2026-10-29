"""Experiment configuration.

Each pipeline configuration is a YAML file that may ``extends:`` another file. Values are
deep-merged, so ``configs/composed.yaml`` only states what differs from ``configs/base.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


@dataclass
class LadderRung:
    model: str
    max_score: float


@dataclass
class CachingConfig:
    enabled: bool = False
    threshold: float = 0.92
    scope: str = "agent"  # "agent" | "pipeline"
    model_aware: bool = False  # True: a hit requires the same model that produced the entry
    backend: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379/0"


@dataclass
class BanditConfig:
    alpha: float = 1.0  # LinUCB exploration width
    ridge: float = 1.0  # L2 prior on each arm's linear model
    cost_weight: float = 0.5  # lambda in reward = success - lambda * cost / reference_cost


@dataclass
class RoutingConfig:
    enabled: bool = False
    default_model: str = "gpt-4o"
    threshold: float = 0.5  # below = low tier, at/above = high tier
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "query_tokens": 0.25,
            "tool_calls": 0.25,
            "context_tokens": 0.25,
            "dependency_depth": 0.25,
        }
    )
    norms: dict[str, float] = field(
        default_factory=lambda: {
            "query_tokens": 512,
            "tool_calls": 4,
            "context_tokens": 8000,
            "dependency_depth": 3,
        }
    )
    ladder: list[LadderRung] = field(default_factory=list)
    policy: str = "heuristic"  # "heuristic" | "linucb"
    bandit: BanditConfig = field(default_factory=BanditConfig)


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    threshold: float = 0.88
    self_report_weight: float = 0.5  # rest goes to consistency between consecutive drafts


@dataclass
class AgentConfig:
    max_iterations: int = 5
    top_k: int = 4
    max_subtasks: int = 4
    temperature: float = 0.2


@dataclass
class ExperimentConfig:
    name: str
    caching: CachingConfig = field(default_factory=CachingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    embedding_model: str = "text-embedding-3-small"
    reference_model: str = "gpt-4o"  # model whose prices define "GPT-4o-equivalent" cost
    prices: dict[str, ModelPrice] = field(default_factory=dict)

    @property
    def techniques(self) -> dict[str, bool]:
        return {
            "caching": self.caching.enabled,
            "routing": self.routing.enabled,
            "early_stopping": self.early_stopping.enabled,
        }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"circular 'extends' at {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(_load_raw(path.parent / parent, seen), data)
    return data


def from_dict(data: dict[str, Any]) -> ExperimentConfig:
    if "name" not in data:
        raise ValueError("config needs a 'name'")
    routing = dict(data.get("routing", {}))
    routing["ladder"] = [LadderRung(**r) for r in routing.get("ladder", [])]
    routing["bandit"] = BanditConfig(**routing.get("bandit", {}))
    prices = {m: ModelPrice(**p) for m, p in data.get("prices", {}).items()}
    cfg = ExperimentConfig(
        name=data["name"],
        caching=CachingConfig(**data.get("caching", {})),
        routing=RoutingConfig(**routing),
        early_stopping=EarlyStoppingConfig(**data.get("early_stopping", {})),
        agents=AgentConfig(**data.get("agents", {})),
        embedding_model=data.get("embedding_model", "text-embedding-3-small"),
        reference_model=data.get("reference_model", "gpt-4o"),
        prices=prices,
    )
    validate(cfg)
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    return from_dict(_load_raw(Path(path)))


def validate(cfg: ExperimentConfig) -> None:
    if cfg.caching.scope not in {"agent", "pipeline"}:
        raise ValueError(f"caching.scope must be 'agent' or 'pipeline', got {cfg.caching.scope!r}")
    if cfg.caching.backend not in {"memory", "redis"}:
        raise ValueError(
            f"caching.backend must be 'memory' or 'redis', got {cfg.caching.backend!r}"
        )
    if not 0.0 < cfg.caching.threshold <= 1.0:
        raise ValueError("caching.threshold must be in (0, 1]")
    if cfg.routing.enabled:
        if not cfg.routing.ladder:
            raise ValueError("routing.enabled requires a non-empty routing.ladder")
        scores = [r.max_score for r in cfg.routing.ladder]
        if scores != sorted(scores) or scores[-1] < 1.0:
            raise ValueError("routing.ladder max_score must ascend and end at >= 1.0")
    if cfg.routing.policy not in {"heuristic", "linucb"}:
        raise ValueError(
            f"routing.policy must be 'heuristic' or 'linucb', got {cfg.routing.policy!r}"
        )
    b = cfg.routing.bandit
    if b.alpha < 0 or b.ridge <= 0 or b.cost_weight < 0:
        raise ValueError("routing.bandit needs alpha >= 0, ridge > 0, cost_weight >= 0")
    w = cfg.routing.weights
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError("routing.weights must sum to 1")
    if not 0.0 <= cfg.early_stopping.self_report_weight <= 1.0:
        raise ValueError("early_stopping.self_report_weight must be in [0, 1]")
    models = {cfg.routing.default_model, cfg.reference_model, cfg.embedding_model}
    models |= {r.model for r in cfg.routing.ladder}
    missing = sorted(m for m in models if m not in cfg.prices)
    if missing:
        raise ValueError(f"no price entry for: {', '.join(missing)}")
