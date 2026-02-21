"""
GoVec Python Client Library

A Python client for interacting with the GoVec vector database API.

Example:
    >>> from govec_client import GoVecClient, VectorPayload, SparseVector
    >>>
    >>> # Create client
    >>> client = GoVecClient(base_url="http://localhost:8000")
    >>>
    >>> # Insert vector
    >>> client.insert("vec1", [0.1, 0.2, 0.3], metadata={"label": "example"})
    >>>
    >>> # Search
    >>> results = client.search([0.1, 0.2, 0.3], k=5)
    >>> for result in results:
    ...     print(f"{result.id}: {result.score}")
"""

from .client import GoVecClient
from .config import GoVecConfig
from .models import (
    VectorPayload,
    SearchResult,
    SearchRequest,
    InsertResponse,
    DeleteResponse,
    SparseVector
)
from .exceptions import (
    GoVecError,
    ConnectionError,
    ValidationError,
    NotFoundError,
    ServerError,
    TimeoutError
)

__version__ = "0.1.0"

__all__ = [
    "GoVecClient",
    "GoVecConfig",
    "VectorPayload",
    "SearchResult",
    "SearchRequest",
    "InsertResponse",
    "DeleteResponse",
    "SparseVector",
    "GoVecError",
    "ConnectionError",
    "ValidationError",
    "NotFoundError",
    "ServerError",
    "TimeoutError",
]
