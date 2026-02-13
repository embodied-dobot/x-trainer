"""
Utility functions for the teleoperation system.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def generate_ssl_certificates(cert_path: str = "cert.pem", key_path: str = "key.pem") -> bool:
    """
    Automatically generate self-signed SSL certificates if they don't exist.
    
    Args:
        cert_path: Path where to save the certificate file
        key_path: Path where to save the private key file
        
    Returns:
        True if certificates exist or were generated successfully, False otherwise
    """
    # Check if certificates already exist
    if os.path.exists(cert_path) and os.path.exists(key_path):
        logger.info(f"SSL certificates already exist: {cert_path}, {key_path}")
        return True
    
    logger.info("SSL certificates not found, generating self-signed certificates...")
    
    try:
        # Generate self-signed certificate using openssl
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path,
            "-out", cert_path,
            "-sha256", "-days", "365", "-nodes",
            "-subj", "/C=US/ST=Test/L=Test/O=Test/OU=Test/CN=localhost"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Set appropriate permissions (readable by owner only for security)
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
        
        logger.info(f"SSL certificates generated successfully: {cert_path}, {key_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate SSL certificates: {e}")
        logger.error(f"Command output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("OpenSSL not found. Please install OpenSSL to generate certificates.")
        logger.error("On Ubuntu/Debian: sudo apt-get install openssl")
        logger.error("On macOS: brew install openssl")
        return False
    except Exception as e:
        logger.error(f"Unexpected error generating SSL certificates: {e}")
        return False

def ensure_ssl_certificates(cert_path: str = "cert.pem", key_path: str = "key.pem") -> bool:
    """
    Ensure SSL certificates exist, generating them if necessary.
    
    Args:
        cert_path: Path to certificate file
        key_path: Path to private key file
        
    Returns:
        True if certificates are available, False if generation failed
    """
    if not generate_ssl_certificates(cert_path, key_path):
        logger.error("Could not ensure SSL certificates are available")
        logger.error("Manual certificate generation may be required:")
        logger.error("openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj \"/C=US/ST=Test/L=Test/O=Test/OU=Test/CN=localhost\"")
        return False
    
    return True 