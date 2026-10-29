import os

import numpy as np

from costcompose.cache import RedisBackend


def test_redis_backend_roundtrip():
    url = os.environ.get("REDIS_URL")
    if not url:
        return  # runs in CI's "full" job, which starts a Redis service
    b = RedisBackend(url, prefix="costcompose:test")
    b.clear()
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert b.nearest("ns", v) is None
    b.add("ns", v, {"x": 1})
    sim, payload = b.nearest("ns", v)
    assert abs(sim - 1.0) < 1e-6 and payload == {"x": 1}
    b.clear()
    assert b.nearest("ns", v) is None
