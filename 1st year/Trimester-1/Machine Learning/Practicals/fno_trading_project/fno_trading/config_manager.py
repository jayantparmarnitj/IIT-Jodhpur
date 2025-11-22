
import json
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    def __init__(self, path: str = "config/config.json"):
        self.path = Path(path)
        self._cfg = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config file {self.path} not found")
        return json.loads(self.path.read_text())

    def get(self, key: str, default=None):
        keys = key.split(".")
        cur = self._cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def as_dict(self):
        return self._cfg
