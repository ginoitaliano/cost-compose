"""Model routing.

Two policies share one interface:

``heuristic`` (the paper's router). Each subtask is scored on four features (query length,
tool-invocation count, context size, inter-agent dependency depth), each capped to [0, 1] and
combined with weights that sum to 1. The score picks a rung on a model ladder ordered cheapest
to most capable; rungs below ``threshold`` form the low tier. Nothing is learned.

``linucb`` (learned). A disjoint LinUCB contextual bandit over the same four features, one arm
per ladder model. It learns online, task by task, from

    reward = 1[task succeeded] - cost_weight * (call cost / same tokens at reference price)

Every routing decision in a task shares the task's success signal (the pipeline only observes
success at the end) but gets its own cost term, so cost credit is per call. Decisions answered
from the cache are not used for updates: the chosen model never ran, so its cost and quality
were not observed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import RoutingConfig

FEATURES = ("query_tokens", "tool_calls", "context_tokens", "dependency_depth")


@dataclass(frozen=True)
class ComplexityFeatures:
    query_tokens: int
    tool_calls: int
    context_tokens: int
    dependency_depth: int


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    tier: str  # "low" | "high" | "fixed"
    score: float
    arm: int = -1
    context: tuple[float, ...] = field(default=(), compare=False)


class Router:
    """Shared base. Learning hooks are no-ops for policies that don't learn."""

    policy = "base"

    def __init__(self, cfg: RoutingConfig) -> None:
        self.cfg = cfg
        self.distribution: Counter = Counter()

    def normalised(self, f: ComplexityFeatures) -> np.ndarray:
        values = asdict(f)
        return np.array([min(values[n] / self.cfg.norms[n], 1.0) for n in FEATURES])

    def score(self, f: ComplexityFeatures) -> float:
        weights = np.array([self.cfg.weights[n] for n in FEATURES])
        return float(weights @ self.normalised(f))

    def _tier(self, model: str) -> str:
        low = {r.model for r in self.cfg.ladder if r.max_score <= self.cfg.threshold}
        return "low" if model in low else "high"

    def route(self, f: ComplexityFeatures) -> RoutingDecision:
        if not self.cfg.enabled:
            decision = RoutingDecision(self.cfg.default_model, "fixed", self.score(f))
        else:
            decision = self._choose(f)
        self.distribution[decision.model] += 1
        return decision

    def _choose(self, f: ComplexityFeatures) -> RoutingDecision:
        raise NotImplementedError

    def observe(self, decision: RoutingDecision, cost_usd: float, reference_usd: float) -> None:
        """Record a decision whose model actually ran (not a cache hit)."""

    def end_task(self, success: bool) -> int:
        """Apply this task's outcome. Returns the number of updates made."""
        return 0

    def reset(self) -> None:
        self.distribution = Counter()

    def state_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "distribution": dict(self.distribution)}


class ComplexityRouter(Router):
    policy = "heuristic"

    def _choose(self, f: ComplexityFeatures) -> RoutingDecision:
        s = self.score(f)
        ladder = self.cfg.ladder
        rung = next((r for r in ladder if s < r.max_score), ladder[-1])
        tier = "low" if s < self.cfg.threshold else "high"
        return RoutingDecision(rung.model, tier, s)


class LinUCBRouter(Router):
    """Disjoint LinUCB (Li et al., 2010) over [1, 4 normalised complexity features]."""

    policy = "linucb"

    def __init__(self, cfg: RoutingConfig) -> None:
        super().__init__(cfg)
        self.models = [r.model for r in cfg.ladder]
        self.dim = 1 + len(FEATURES)
        self._init_params()

    def _init_params(self) -> None:
        k, d, ridge = len(self.models), self.dim, self.cfg.bandit.ridge
        self.A = np.stack([ridge * np.eye(d) for _ in range(k)])
        self.b = np.zeros((k, d))
        self.pulls = np.zeros(k, dtype=int)
        self.reward_sum = np.zeros(k)
        self._pending: list[tuple[int, np.ndarray, float]] = []

    def context(self, f: ComplexityFeatures) -> np.ndarray:
        return np.concatenate([[1.0], self.normalised(f)])

    def ucb(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.models))
        for a in range(len(self.models)):
            a_inv = np.linalg.inv(self.A[a])
            theta = a_inv @ self.b[a]
            out[a] = theta @ x + self.cfg.bandit.alpha * np.sqrt(x @ a_inv @ x)
        return out

    def _choose(self, f: ComplexityFeatures) -> RoutingDecision:
        x = self.context(f)
        arm = int(np.argmax(self.ucb(x)))  # ties go to the cheaper (earlier) arm
        model = self.models[arm]
        return RoutingDecision(model, self._tier(model), self.score(f), arm, tuple(x))

    def observe(self, decision, cost_usd, reference_usd):
        if decision.arm < 0:
            return
        ratio = cost_usd / reference_usd if reference_usd > 0 else 0.0
        self._pending.append((decision.arm, np.array(decision.context), ratio))

    def end_task(self, success: bool) -> int:
        lam = self.cfg.bandit.cost_weight
        for arm, x, ratio in self._pending:
            r = float(success) - lam * ratio
            self.A[arm] += np.outer(x, x)
            self.b[arm] += r * x
            self.pulls[arm] += 1
            self.reward_sum[arm] += r
        n = len(self._pending)
        self._pending = []
        return n

    def reset(self) -> None:
        super().reset()
        self._init_params()

    def state_dict(self) -> dict[str, Any]:
        thetas = [np.linalg.solve(self.A[a], self.b[a]).tolist() for a in range(len(self.models))]
        return {
            **super().state_dict(),
            "features": ["bias", *FEATURES],
            "arms": [
                {
                    "model": m,
                    "pulls": int(self.pulls[a]),
                    "mean_reward": float(self.reward_sum[a] / self.pulls[a])
                    if self.pulls[a] else None,
                    "theta": thetas[a],
                }
                for a, m in enumerate(self.models)
            ],
        }


def make_router(cfg: RoutingConfig) -> Router:
    if cfg.enabled and cfg.policy == "linucb":
        return LinUCBRouter(cfg)
    return ComplexityRouter(cfg)
