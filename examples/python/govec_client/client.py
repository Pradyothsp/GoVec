"""
Main GoVec client implementation.

This module provides the GoVecClient class for interacting with the GoVec vector database API.
"""

import time
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import GoVecConfig
from .models import (
    VectorPayload,
    SearchRequest,
    SearchResult,
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


class GoVecClient:
    """
    Client for interacting with GoVec vector database API.

    This client provides methods for inserting, searching, and deleting vectors
    with automatic retry logic and connection pooling.

    Example:
        >>> client = GoVecClient(base_url="http://localhost:8000")
        >>> client.insert("vec1", [0.1, 0.2, 0.3], metadata={"label": "example"})
        >>> results = client.search([0.1, 0.2, 0.3], k=5)

    Or using context manager:
        >>> with GoVecClient() as client:
        ...     client.insert("vec1", [0.1, 0.2, 0.3])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        config: Optional[GoVecConfig] = None
    ):
        """
        Initialize GoVec client.

        Args:
            base_url: Base URL for GoVec API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_backoff: Backoff factor for retries
            config: Optional GoVecConfig instance (overrides other parameters)
        """
        if config is not None:
            self.config = config
        else:
            self.config = GoVecConfig(
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff
            )

        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Create requests session with connection pooling and retry logic.

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Configure retry strategy for transient failures
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[500, 502, 503, 504],  # Retry only on server errors
            allowed_methods=["GET", "POST", "DELETE"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self.close()

    def close(self):
        """Close the HTTP session."""
        if self.session:
            self.session.close()

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON data from response

        Raises:
            ValidationError: For 400 Bad Request
            NotFoundError: For 404 Not Found
            ServerError: For 5xx errors
            GoVecError: For other errors
        """
        try:
            data = response.json()
        except ValueError:
            raise GoVecError(f"Invalid JSON response: {response.text}", response.status_code)

        # Check HTTP status code
        if response.status_code == 400:
            error_msg = data.get("error", "Validation failed")
            raise ValidationError(error_msg, 400)
        elif response.status_code == 404:
            error_msg = data.get("error", "Resource not found")
            raise NotFoundError(error_msg, 404)
        elif response.status_code >= 500:
            error_msg = data.get("error", "Internal server error")
            raise ServerError(error_msg, response.status_code)
        elif not response.ok:
            error_msg = data.get("error", f"Request failed with status {response.status_code}")
            raise GoVecError(error_msg, response.status_code)

        # Check success field in response envelope
        if not data.get("success", False):
            error_msg = data.get("error", "Operation failed")
            raise GoVecError(error_msg)

        return data.get("data", {})

    def health_check(self) -> bool:
        """
        Check if GoVec server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=self.config.timeout
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def insert(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[SparseVector] = None
    ) -> InsertResponse:
        """
        Insert a vector into the database.

        Args:
            id: Unique identifier for the vector
            vector: Dense vector (list of floats)
            metadata: Optional metadata dictionary
            sparse_vector: Optional sparse vector for hybrid search

        Returns:
            InsertResponse with operation status

        Raises:
            ValidationError: If request validation fails
            ConnectionError: If connection fails
            GoVecError: For other errors
        """
        # Wrap payload creation to catch validation errors
        try:
            payload = VectorPayload(
                id=id,
                vector=vector,
                metadata=metadata,
                sparse_vector=sparse_vector
            )
        except ValueError as e:
            raise ValidationError(str(e), 400)

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/v1/vectors",
                json=payload.to_dict(),
                timeout=self.config.timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out", None)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection failed: {str(e)}", None)
        except requests.RequestException as e:
            raise GoVecError(f"Request failed: {str(e)}", None)

        data = self._handle_response(response)
        return InsertResponse.from_api_response(data)

    def insert_batch(self, vectors: List[VectorPayload]) -> List[InsertResponse]:
        """
        Insert multiple vectors efficiently.

        Args:
            vectors: List of VectorPayload objects

        Returns:
            List of InsertResponse objects
        """
        responses = []
        for payload in vectors:
            response = self.insert(
                id=payload.id,
                vector=payload.vector,
                metadata=payload.metadata,
                sparse_vector=payload.sparse_vector
            )
            responses.append(response)
        return responses

    def search(
        self,
        vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[SparseVector] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector (list of floats)
            k: Number of results to return (top-K)
            filters: Optional metadata filters
            sparse_vector: Optional sparse vector for hybrid search

        Returns:
            List of SearchResult objects, sorted by similarity (descending)

        Raises:
            ValidationError: If request validation fails
            ConnectionError: If connection fails
            GoVecError: For other errors
        """
        # Wrap request creation to catch validation errors
        try:
            request = SearchRequest(
                vector=vector,
                k=k,
                filters=filters,
                sparse_vector=sparse_vector
            )
        except ValueError as e:
            raise ValidationError(str(e), 400)

        try:
            response = self.session.post(
                f"{self.config.base_url}/api/v1/vectors/search",
                json=request.to_dict(),
                timeout=self.config.timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out", None)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection failed: {str(e)}", None)
        except requests.RequestException as e:
            raise GoVecError(f"Request failed: {str(e)}", None)

        data = self._handle_response(response)

        # Handle both single dict and list responses
        if isinstance(data, dict):
            results_data = data.get("results", [])
        elif isinstance(data, list):
            results_data = data
        else:
            results_data = []

        return [SearchResult.from_api_response(item) for item in results_data]

    def delete(self, id: str) -> DeleteResponse:
        """
        Delete a vector from the database.

        Args:
            id: Vector ID to delete

        Returns:
            DeleteResponse with operation status

        Raises:
            NotFoundError: If vector not found
            ConnectionError: If connection fails
            GoVecError: For other errors
        """
        try:
            response = self.session.delete(
                f"{self.config.base_url}/api/v1/vectors/{id}",
                timeout=self.config.timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out", None)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Connection failed: {str(e)}", None)
        except requests.RequestException as e:
            raise GoVecError(f"Request failed: {str(e)}", None)

        data = self._handle_response(response)
        return DeleteResponse.from_api_response(data)
