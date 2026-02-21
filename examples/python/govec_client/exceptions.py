"""
Custom exceptions for GoVec client library.

This module defines a hierarchy of exceptions for different error scenarios
when interacting with the GoVec vector database API.
"""


class GoVecError(Exception):
    """Base exception for all GoVec client errors."""

    def __init__(self, message: str, status_code: int = None):
        """
        Initialize GoVec error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code (if applicable)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class ConnectionError(GoVecError):
    """Connection to GoVec server failed."""
    pass


class ValidationError(GoVecError):
    """Request validation failed (400 Bad Request)."""
    pass


class NotFoundError(GoVecError):
    """Requested resource not found (404 Not Found)."""
    pass


class ServerError(GoVecError):
    """Internal server error (5xx)."""
    pass


class TimeoutError(GoVecError):
    """Request timed out."""
    pass
