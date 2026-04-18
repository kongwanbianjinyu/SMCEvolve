"""Event logger that writes both a JSONL stream and a browsable file tree.

Layout under the run directory:

  events.jsonl                              one record per line (machine-readable)
  event_logs/
    _final.json
    island_0/
      _init.json
      iter_001/
        _summary.json                       SMC step summary (λ, Δβ, β_t, ESS, ...)
        p0_k0/                              one proposal (particle 0, proposal 0)
          prompt.txt
          response.txt
          program.py
          info.json
        p0_k1/
        p1_k0/
        ...
      iter_002/
        ...
    island_1/
      ...

Both outputs are written under a single thread lock so the logger is safe to
call from `asyncio.gather` workers and `asyncio.to_thread` threads.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tree_root = self.run_dir / "event_logs"
        self.tree_root.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "events.jsonl"
        self._fp = open(self.jsonl_path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.jsonl_path

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_json_default, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._write_tree(record)

    def close(self) -> None:
        with self._lock:
            if not self._fp.closed:
                self._fp.close()

    # ------------------------------------------------------------------ tree

    def _write_tree(self, r: dict[str, Any]) -> None:
        rtype = r.get("type")
        if rtype == "proposal":
            self._write_proposal(r)
        elif rtype == "resample":
            self._write_resample(r)
        elif rtype == "step_summary":
            self._write_step_summary(r)
        elif rtype == "init":
            self._write_init(r)
        elif rtype == "final":
            self._write_final(r)

    def _write_resample(self, r: dict[str, Any]) -> None:
        d = (
            self.tree_root
            / f"island_{r['island']}"
            / f"iter_{int(r['iteration']):03d}"
        )
        d.mkdir(parents=True, exist_ok=True)
        _dump_json(d / "_resample.json", r)

    def _write_proposal(self, r: dict[str, Any]) -> None:
        d = (
            self.tree_root
            / f"island_{r['island']}"
            / f"iter_{int(r['iteration']):03d}"
            / f"p{r['particle_idx']}_k{r['k_step']}"
        )
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompt.txt").write_text(r.get("prompt") or "", encoding="utf-8")
        (d / "response.txt").write_text(r.get("response") or "", encoding="utf-8")
        (d / "program.py").write_text(r.get("program") or "", encoding="utf-8")
        info = {k: v for k, v in r.items() if k not in ("prompt", "response", "program")}
        _dump_json(d / "info.json", info)

    def _write_step_summary(self, r: dict[str, Any]) -> None:
        d = (
            self.tree_root
            / f"island_{r['island']}"
            / f"iter_{int(r['iteration']):03d}"
        )
        d.mkdir(parents=True, exist_ok=True)
        _dump_json(d / "_summary.json", r)

    def _write_init(self, r: dict[str, Any]) -> None:
        d = self.tree_root / f"island_{r['island']}"
        d.mkdir(parents=True, exist_ok=True)
        _dump_json(d / "_init.json", r)

    def _write_final(self, r: dict[str, Any]) -> None:
        _dump_json(self.tree_root / "_final.json", r)


def _dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return repr(obj)
