from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Particle:
    program: str
    reward: float = float("-inf")
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict[str, Any] = field(default_factory=dict)
