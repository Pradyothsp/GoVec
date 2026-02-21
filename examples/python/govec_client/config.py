"""
Configuration management for GoVec client.

This module provides configuration options with environment variable overrides
following the 12-factor app pattern.
"""

import os
from dataclasses import dataclass


@dataclass
class GoVecConfig:
    """
    Configuration for GoVec client.

    Attributes:
        base_url: Base URL for GoVec API (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_backoff: Backoff factor for retries in seconds (default: 0.5)
    """
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 0.5

    @classmethod
    def from_env(cls) -> 'GoVecConfig':
        """
        Create configuration from environment variables.

        Environment variables:
            GOVEC_BASE_URL: Base URL for API
            GOVEC_TIMEOUT: Request timeout in seconds
            GOVEC_MAX_RETRIES: Maximum retry attempts
            GOVEC_RETRY_BACKOFF: Retry backoff factor

        Returns:
            GoVecConfig instance with values from environment or defaults
        """
        return cls(
            base_url=os.getenv("GOVEC_BASE_URL", "http://localhost:8000"),
            timeout=int(os.getenv("GOVEC_TIMEOUT", "30")),
            max_retries=int(os.getenv("GOVEC_MAX_RETRIES", "3")),
            retry_backoff=float(os.getenv("GOVEC_RETRY_BACKOFF", "0.5"))
        )

    def __post_init__(self):
        """Validate configuration values."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive.")

        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative.")

        if self.retry_backoff < 0:
            raise ValueError("Retry backoff cannot be negative.")

        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip("/")
