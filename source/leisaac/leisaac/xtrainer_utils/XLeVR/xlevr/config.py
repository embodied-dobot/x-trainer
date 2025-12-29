"""
Configuration module for the unified teleoperation system.
Loads configuration from config.yaml file with fallback to default values.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Default configuration values (fallback if YAML file doesn't exist)
DEFAULT_CONFIG = {
    "network": {
        "https_port": 8443,
        "websocket_port": 8442,
        "host_ip": "0.0.0.0"
    },
    "ssl": {
        "certfile": "cert.pem",
        "keyfile": "key.pem"
    },
    "robot": {
        "left_arm": {
            "name": "Left Arm",
            "port": "/dev/ttyACM0",
            "enabled": True
        },
        "right_arm": {
            "name": "Right Arm",
            "port": "/dev/ttyACM1",
            "enabled": True
        },
        "vr_to_robot_scale": 1.0,
        "send_interval": 0.05,
    },
    "control": {
        "keyboard": {
            "pos_step": 0.01,
            "angle_step": 5.0
        }
    }
}

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with fallback to defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Deep merge yaml config into default config
                    _deep_merge(config, yaml_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration")
    else:
        print(f"Config file {config_path} not found, using defaults")
    
    return config

def save_config(config: dict, config_path: str = "config.yaml"):
    """Save configuration to YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False

def _deep_merge(base: dict, update: dict):
    """Deep merge update dict into base dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

# Load configuration
_config_data = load_config()

# Extract values for backward compatibility
HTTPS_PORT = _config_data["network"]["https_port"]
WEBSOCKET_PORT = _config_data["network"]["websocket_port"]
HOST_IP = _config_data["network"]["host_ip"]

CERTFILE = _config_data["ssl"]["certfile"]
KEYFILE = _config_data["ssl"]["keyfile"]

VR_TO_ROBOT_SCALE = _config_data["robot"]["vr_to_robot_scale"]
SEND_INTERVAL = _config_data["robot"]["send_interval"]

POS_STEP = _config_data["control"]["keyboard"]["pos_step"]
ANGLE_STEP = _config_data["control"]["keyboard"]["angle_step"]

# Device Ports
DEFAULT_FOLLOWER_PORTS = {
    "left": _config_data["robot"]["left_arm"]["port"],
    "right": _config_data["robot"]["right_arm"]["port"]
}

@dataclass
class XLeVRConfig:
    """Main configuration class for the teleoperation system (VR-only)."""
    # Network settings
    https_port: int = HTTPS_PORT
    websocket_port: int = WEBSOCKET_PORT
    host_ip: str = HOST_IP
    # SSL settings
    certfile: str = CERTFILE
    keyfile: str = KEYFILE
    # Device ports (optional, for compatibility)
    follower_ports: Dict[str, str] = None
    # Control flags (only VR and HTTPS relevant)
    enable_vr: bool = True
    enable_keyboard: bool = False
    enable_https: bool = True
    log_level: str = "warning"
    vr_to_robot_scale: float = VR_TO_ROBOT_SCALE
    # Optionally, webapp_dir if used elsewhere
    webapp_dir: str = "webapp"
    def __post_init__(self):
        if self.follower_ports is None:
            self.follower_ports = {
                "left": _config_data["robot"]["left_arm"]["port"],
                "right": _config_data["robot"]["right_arm"]["port"]
            }
    @property
    def ssl_files_exist(self) -> bool:
        return os.path.exists(self.certfile) and os.path.exists(self.keyfile)
    def ensure_ssl_certificates(self) -> bool:
        from .utils import ensure_ssl_certificates
        return ensure_ssl_certificates(self.certfile, self.keyfile)
    @property
    def webapp_exists(self) -> bool:
        return os.path.exists(self.webapp_dir)

def get_config_data():
    """Get the current configuration data."""
    return _config_data.copy()

def update_config_data(new_config: dict):
    """Update the configuration data and save to file."""
    global _config_data
    _deep_merge(_config_data, new_config)
    return save_config(_config_data)

# Global configuration instance
config = XLeVRConfig() 
