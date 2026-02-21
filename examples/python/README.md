# GoVec Python Client

A Python client library for interacting with the [GoVec](https://github.com/yourorg/govec) vector database API.

## Features

- 🚀 **Simple API** - Intuitive methods for vector operations
- 🔄 **Connection Pooling** - Automatic connection reuse for better performance
- 🔁 **Retry Logic** - Automatic retries with exponential backoff for transient failures
- 🛡️ **Type Safety** - Full type hints for IDE autocomplete and type checking
- 📦 **Hybrid Search** - Support for dense + sparse vectors (semantic + keyword matching)
- ⚡ **Batch Operations** - Efficient bulk insertion and search
- 🐍 **Pythonic** - Follows Python best practices and PEP 8

## Installation

### Prerequisites

- Python 3.8 or higher
- GoVec server running (see [GoVec documentation](../../README.md))

### Install Dependencies

```bash
cd examples/python
pip install -r requirements.txt
```

Or install directly:

```bash
pip install requests>=2.31.0
```

## Quick Start

### 1. Start GoVec Server

```bash
# From the GoVec project root
cd /Volumes/work/govec
task run
```

The server will start on `http://localhost:8000` by default.

### 2. Basic Usage

```python
from govec_client import GoVecClient

# Create client
client = GoVecClient(base_url="http://localhost:8000")

# Insert a vector
client.insert(
    id="vec1",
    vector=[0.1, 0.2, 0.3, 0.4],
    metadata={"label": "example", "category": "test"}
)

# Search for similar vectors
results = client.search(vector=[0.1, 0.2, 0.3, 0.4], k=5)

for result in results:
    print(f"{result.id}: {result.score:.4f}")
    print(f"  Metadata: {result.metadata}")

# Clean up
client.close()
```

### 3. Using Context Manager (Recommended)

```python
from govec_client import GoVecClient

with GoVecClient() as client:
    # Health check
    if not client.health_check():
        print("Server is not healthy!")
        return

    # Insert vector
    response = client.insert("vec1", [0.1, 0.2, 0.3, 0.4])
    print(f"Status: {response.status}")

    # Search
    results = client.search([0.1, 0.2, 0.3, 0.4], k=10)
    for result in results:
        print(f"{result.id}: {result.score:.4f}")

# Session automatically closed
```

## Example Scripts

The `examples/` directory contains complete, runnable examples:

### 1. Insert Vectors (SAVING)

```bash
python examples/01_insert_vectors.py
```

Demonstrates:
- Single vector insertion with metadata
- Sparse vector insertion (hybrid search)
- Batch insertion for efficiency
- Error handling

### 2. Search Vectors (RETRIEVING)

```bash
python examples/02_search_vectors.py
```

Demonstrates:
- Basic similarity search
- Top-K retrieval
- Metadata filtering
- Hybrid search (dense + sparse vectors)
- Result iteration and display

### 3. Hybrid Search

```bash
python examples/03_hybrid_search.py
```

Demonstrates:
- BM25 sparse vectors with token IDs
- Combined dense + sparse scoring
- Real-world hybrid search use cases

### 4. Batch Operations

```bash
python examples/04_batch_operations.py
```

Demonstrates:
- Efficient batch insertion
- Batch search operations
- Performance metrics

## API Reference

### GoVecClient

Main client class for interacting with GoVec.

#### Constructor

```python
GoVecClient(
    base_url: str = "http://localhost:8000",
    timeout: int = 30,
    max_retries: int = 3,
    retry_backoff: float = 0.5,
    config: Optional[GoVecConfig] = None
)
```

**Parameters:**
- `base_url`: GoVec server URL
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum retry attempts for transient failures
- `retry_backoff`: Backoff factor for exponential retry delay
- `config`: Optional `GoVecConfig` instance (overrides other parameters)

#### Methods

##### `insert(id, vector, metadata=None, sparse_vector=None)`

Insert a vector into the database.

```python
response = client.insert(
    id="vec1",
    vector=[0.1, 0.2, 0.3, 0.4],
    metadata={"label": "example"},
    sparse_vector=SparseVector(indices=[1, 5, 10], values=[0.8, 0.6, 0.4])
)
```

**Parameters:**
- `id` (str): Unique identifier
- `vector` (List[float]): Dense vector
- `metadata` (Optional[Dict]): Associated metadata
- `sparse_vector` (Optional[SparseVector]): Sparse vector for hybrid search

**Returns:** `InsertResponse` with status and success flag

**Raises:**
- `ValidationError`: Invalid request (400)
- `ConnectionError`: Connection failed
- `TimeoutError`: Request timeout

##### `search(vector, k=10, filters=None, sparse_vector=None)`

Search for similar vectors.

```python
results = client.search(
    vector=[0.1, 0.2, 0.3, 0.4],
    k=5,
    sparse_vector=SparseVector(indices=[1, 5], values=[0.9, 0.7])
)
```

**Parameters:**
- `vector` (List[float]): Query vector
- `k` (int): Number of results (default: 10)
- `filters` (Optional[Dict]): Metadata filters
- `sparse_vector` (Optional[SparseVector]): Sparse query vector

**Returns:** `List[SearchResult]` sorted by similarity (descending)

##### `insert_batch(vectors)`

Insert multiple vectors efficiently.

```python
from govec_client import VectorPayload

vectors = [
    VectorPayload(id="vec1", vector=[0.1, 0.2, 0.3, 0.4]),
    VectorPayload(id="vec2", vector=[0.5, 0.6, 0.7, 0.8]),
]
responses = client.insert_batch(vectors)
```

**Parameters:**
- `vectors` (List[VectorPayload]): List of vectors to insert

**Returns:** `List[InsertResponse]`

##### `delete(id)`

Delete a vector from the database.

```python
response = client.delete("vec1")
```

**Parameters:**
- `id` (str): Vector ID to delete

**Returns:** `DeleteResponse` with status and success flag

**Raises:**
- `NotFoundError`: Vector not found (404)

##### `health_check()`

Check if server is healthy.

```python
if client.health_check():
    print("Server is healthy")
```

**Returns:** `bool` - True if healthy, False otherwise

### Data Models

#### SparseVector

Sparse vector representation for hybrid search.

```python
from govec_client import SparseVector

sparse = SparseVector(
    indices=[10, 25, 100],  # Token IDs (must be sorted ascending)
    values=[0.8, 0.6, 0.4]  # BM25 weights
)
```

**Attributes:**
- `indices` (List[int]): Token IDs (must be sorted ascending, non-negative)
- `values` (List[float]): Weights corresponding to each index

**Methods:**
- `is_valid()`: Check if sparse vector is valid
- `is_empty()`: Check if sparse vector is empty
- `to_dict()`: Convert to dictionary for JSON serialization

#### VectorPayload

Vector insertion payload.

```python
from govec_client import VectorPayload

payload = VectorPayload(
    id="vec1",
    vector=[0.1, 0.2, 0.3, 0.4],
    metadata={"label": "example"},
    sparse_vector=sparse
)
```

**Attributes:**
- `id` (str): Unique identifier
- `vector` (List[float]): Dense vector
- `metadata` (Optional[Dict]): Optional metadata
- `sparse_vector` (Optional[SparseVector]): Optional sparse vector

#### SearchResult

Search result from API.

```python
# Returned from client.search()
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

**Attributes:**
- `id` (str): Vector ID
- `score` (float): Similarity score (higher = more similar)
- `metadata` (Dict): Associated metadata

## Configuration

### Environment Variables

Override configuration using environment variables:

```bash
export GOVEC_BASE_URL="http://localhost:8000"
export GOVEC_TIMEOUT=30
export GOVEC_MAX_RETRIES=3
export GOVEC_RETRY_BACKOFF=0.5
```

### Using GoVecConfig

```python
from govec_client import GoVecClient, GoVecConfig

# Load from environment
config = GoVecConfig.from_env()
client = GoVecClient(config=config)

# Or specify directly
config = GoVecConfig(
    base_url="http://localhost:9000",
    timeout=60,
    max_retries=5
)
client = GoVecClient(config=config)
```

## Error Handling

The client provides a hierarchy of exceptions for different error scenarios:

```python
from govec_client import GoVecClient
from govec_client.exceptions import (
    GoVecError,
    ValidationError,
    NotFoundError,
    ConnectionError,
    ServerError,
    TimeoutError
)

try:
    client.insert("vec1", [0.1, 0.2, 0.3, 0.4])
except ValidationError as e:
    print(f"Validation failed: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except NotFoundError as e:
    print(f"Not found: {e}")
except ServerError as e:
    print(f"Server error: {e}")
except GoVecError as e:
    print(f"General error: {e}")
```

### Exception Hierarchy

- `GoVecError` - Base exception for all errors
  - `ConnectionError` - Connection failed
  - `ValidationError` - Request validation failed (400)
  - `NotFoundError` - Resource not found (404)
  - `ServerError` - Internal server error (5xx)
  - `TimeoutError` - Request timeout

## Advanced Usage

### Hybrid Search

Combine dense (semantic) and sparse (keyword) vectors:

```python
from govec_client import GoVecClient, SparseVector

client = GoVecClient()

# Insert with hybrid vector
client.insert(
    id="doc1",
    vector=[0.8, 0.6, 0.4, 0.2],  # Semantic embedding
    sparse_vector=SparseVector(
        indices=[100, 200, 300],  # Token IDs: "python", "machine", "learning"
        values=[0.9, 0.8, 0.9]    # BM25 scores
    ),
    metadata={"title": "Python Machine Learning"}
)

# Search with hybrid query
results = client.search(
    vector=[0.8, 0.6, 0.4, 0.2],
    sparse_vector=SparseVector(
        indices=[100, 200],  # Match "python", "machine"
        values=[1.0, 0.9]
    ),
    k=10
)
```

**Important:** Sparse vector indices must be:
- Non-negative integers (uint32)
- Sorted in ascending order
- Same length as values array

### Batch Operations

For better performance with multiple vectors:

```python
from govec_client import VectorPayload

# Prepare batch
vectors = [
    VectorPayload(id=f"vec{i}", vector=[i*0.1]*4, metadata={"index": i})
    for i in range(100)
]

# Insert batch
responses = client.insert_batch(vectors)
print(f"Inserted {len(responses)} vectors")

# Multiple searches
queries = [[0.1]*4, [0.5]*4, [0.9]*4]
for query in queries:
    results = client.search(query, k=5)
    print(f"Found {len(results)} results")
```

## Testing

Run the example scripts to verify your installation:

```bash
# Start GoVec server first
cd /Volumes/work/govec && task run

# In another terminal, run examples
cd examples/python
python examples/01_insert_vectors.py
python examples/02_search_vectors.py
python examples/03_hybrid_search.py
python examples/04_batch_operations.py
```

## Troubleshooting

### "Server is not healthy" error

**Cause:** GoVec server is not running or not accessible.

**Solution:**
```bash
# Start the server
cd /Volumes/work/govec
task run
```

### Connection timeout

**Cause:** Server is slow or network issues.

**Solution:**
```python
# Increase timeout
client = GoVecClient(timeout=60)
```

### ValidationError on insert

**Cause:** Invalid vector data (empty vector, mismatched dimensions, etc.).

**Solution:**
- Ensure vector is not empty
- Ensure all vector elements are numeric
- For sparse vectors, ensure indices are sorted and same length as values

### Import errors

**Cause:** Dependencies not installed.

**Solution:**
```bash
pip install -r requirements.txt
```

## Performance Tips

1. **Use batch operations** for bulk data ingestion
2. **Enable connection pooling** (automatic with `requests.Session`)
3. **Adjust retry settings** based on your network conditions
4. **Use context managers** to ensure proper cleanup
5. **Monitor timeout values** for large operations

## License

This client library is part of the GoVec project. See the main project for license information.

## Contributing

Contributions are welcome! Please see the main GoVec project for contribution guidelines.

## Related

- [GoVec Project](../../README.md) - Main vector database documentation
- [GoVec API Documentation](../../build/swagger/) - API reference
- [AGENTS.md](../../AGENTS.md) - Development guide for GoVec

## Support

For issues or questions:
- Open an issue on the GoVec GitHub repository
- Check the GoVec documentation
- Review the example scripts in `examples/`
