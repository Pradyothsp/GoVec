"""
Data models for GoVec client library.

This module defines dataclasses for request/response payloads used by the GoVec API.
All models include type hints and validation logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SparseVector:
    """
    Sparse vector representation for hybrid search.

    Attributes:
        indices: Token IDs (uint32, must be sorted ascending)
        values: BM25 weights corresponding to each index
    """
    indices: List[int]
    values: List[float]

    def __post_init__(self):
        """Validate sparse vector on initialization."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"Indices and values must have same length. "
                f"Got {len(self.indices)} indices and {len(self.values)} values."
            )

        # Check that indices are sorted
        if self.indices != sorted(self.indices):
            raise ValueError("Indices must be sorted in ascending order.")

        # Check that indices are non-negative (uint32)
        if any(idx < 0 for idx in self.indices):
            raise ValueError("Indices must be non-negative (uint32).")

    def is_valid(self) -> bool:
        """Check if sparse vector is valid."""
        return len(self.indices) == len(self.values) and len(self.indices) > 0

    def is_empty(self) -> bool:
        """Check if sparse vector is empty."""
        return len(self.indices) == 0

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for JSON serialization."""
        return {
            "indices": self.indices,
            "values": self.values
        }


@dataclass
class VectorPayload:
    """
    Vector insertion payload.

    Attributes:
        id: Unique identifier for the vector
        vector: Dense vector (list of floats)
        metadata: Optional metadata dictionary
        sparse_vector: Optional sparse vector for hybrid search
    """
    id: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    sparse_vector: Optional[SparseVector] = None

    def __post_init__(self):
        """Validate vector payload on initialization."""
        if not self.id:
            raise ValueError("Vector ID cannot be empty.")

        if not self.vector:
            raise ValueError("Vector cannot be empty.")

        if not all(isinstance(v, (int, float)) for v in self.vector):
            raise ValueError("Vector must contain only numeric values.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        payload = {
            "id": self.id,
            "vector": self.vector
        }

        if self.metadata is not None:
            payload["metadata"] = self.metadata

        if self.sparse_vector is not None:
            payload["sparse_vector"] = self.sparse_vector.to_dict()

        return payload


@dataclass
class SearchResult:
    """
    Search result from GoVec API.

    Attributes:
        id: Vector ID
        score: Similarity score (higher is more similar)
        metadata: Associated metadata
    """
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'SearchResult':
        """
        Create SearchResult from API response.

        The GoVec API uses uppercase field names (ID, Score, Meta),
        this method maps them to Python conventions.

        Args:
            data: Raw API response data

        Returns:
            SearchResult instance
        """
        return cls(
            id=data.get("ID", data.get("id", "")),
            score=data.get("Score", data.get("score", 0.0)),
            metadata=data.get("Meta", data.get("metadata", {}))
        )


@dataclass
class InsertResponse:
    """
    Response from vector insertion.

    Attributes:
        status: Operation status (e.g., "inserted")
        success: Whether operation succeeded
    """
    status: str
    success: bool

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'InsertResponse':
        """Create InsertResponse from API response."""
        return cls(
            status=data.get("status", "unknown"),
            success=data.get("success", False)
        )


@dataclass
class DeleteResponse:
    """
    Response from vector deletion.

    Attributes:
        status: Operation status (e.g., "deleted")
        id: Deleted vector ID
        success: Whether operation succeeded
    """
    status: str
    id: str
    success: bool

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> 'DeleteResponse':
        """Create DeleteResponse from API response."""
        return cls(
            status=data.get("status", "unknown"),
            id=data.get("id", ""),
            success=data.get("success", False)
        )


@dataclass
class SearchRequest:
    """
    Search request payload.

    Attributes:
        vector: Query vector (list of floats)
        k: Number of results to return (top-K)
        filters: Optional metadata filters
        sparse_vector: Optional sparse vector for hybrid search
    """
    vector: List[float]
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    sparse_vector: Optional[SparseVector] = None

    def __post_init__(self):
        """Validate search request on initialization."""
        if not self.vector:
            raise ValueError("Query vector cannot be empty.")

        if self.k <= 0:
            raise ValueError("k must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        payload = {
            "vector": self.vector,
            "k": self.k
        }

        if self.filters is not None:
            payload["filters"] = self.filters

        if self.sparse_vector is not None:
            payload["sparse_vector"] = self.sparse_vector.to_dict()

        return payload
