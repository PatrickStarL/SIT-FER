"""Configuration management for SIT-FER"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """Configuration class for managing experiment settings"""

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional config overrides
        """
        self.config = {}

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Override with kwargs
        self._update_config(self.config, kwargs)

    def _update_config(self, config: Dict, updates: Dict):
        """Recursively update config with new values"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in config:
                self._update_config(config[key], value)
            else:
                config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set config value by dot-notation key"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def to_dict(self) -> Dict:
        """Return configuration as dictionary"""
        return self.config.copy()

    def __repr__(self) -> str:
        return f"Config({yaml.dump(self.config, default_flow_style=False)})"
