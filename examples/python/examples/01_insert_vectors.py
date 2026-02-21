#!/usr/bin/env python3
"""
Example 1: SAVING - Vector Insertion Examples

This script demonstrates how to insert vectors into GoVec with various configurations:
- Single vector insertion with metadata
- Sparse vector insertion (hybrid search)
- Batch insertion for efficiency

Usage:
    python examples/01_insert_vectors.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from govec_client import GoVecClient, SparseVector, VectorPayload
from govec_client.exceptions import GoVecError


def example_1_single_insert(client: GoVecClient):
    """Insert a single vector with metadata."""
    print("\n📥 Example 1: Insert single vector with metadata")
    print("-" * 60)

    try:
        response = client.insert(
            id="example_vec_1",
            vector=[0.1, 0.2, 0.3, 0.4],
            metadata={
                "category": "example",
                "label": "test_vector",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        )
        print(f"✅ Status: {response.status}")
        print(f"   Success: {response.success}")
    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_2_hybrid_insert(client: GoVecClient):
    """Insert vector with sparse vector for hybrid search."""
    print("\n📥 Example 2: Insert vector with sparse vector (hybrid search)")
    print("-" * 60)

    try:
        # Create sparse vector (BM25-like weights for token IDs)
        sparse = SparseVector(
            indices=[10, 25, 100, 500],  # Token IDs (sorted)
            values=[0.8, 0.6, 0.4, 0.2]  # BM25 weights
        )

        response = client.insert(
            id="example_vec_2_hybrid",
            vector=[0.5, 0.25, 0.75, -0.15],
            metadata={"type": "hybrid_example"},
            sparse_vector=sparse
        )
        print(f"✅ Status: {response.status}")
        print(f"   Sparse indices: {sparse.indices}")
        print(f"   Sparse values: {sparse.values}")
    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_3_batch_insert(client: GoVecClient):
    """Insert multiple vectors efficiently using batch operation."""
    print("\n📥 Example 3: Batch insertion")
    print("-" * 60)

    try:
        # Prepare batch of vectors
        vectors = [
            VectorPayload(
                id=f"batch_vec_{i}",
                vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                metadata={"batch": "demo", "index": i}
            )
            for i in range(1, 6)
        ]

        # Insert batch
        responses = client.insert_batch(vectors)

        print(f"✅ Inserted {len(responses)} vectors:")
        for i, response in enumerate(responses):
            print(f"   [{i+1}] {vectors[i].id} - Status: {response.status}")
    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_4_validation_error(client: GoVecClient):
    """Demonstrate validation error handling."""
    print("\n📥 Example 4: Validation error handling")
    print("-" * 60)

    try:
        # This should fail - empty vector
        response = client.insert(
            id="invalid_vec",
            vector=[],  # Empty vector - invalid!
            metadata={"test": "error_handling"}
        )
        print(f"✅ Unexpected success: {response.status}")
    except GoVecError as e:
        print(f"✅ Caught expected error: {e}")


def main():
    """Run all insert examples."""
    print("=" * 60)
    print("GoVec Python Client - SAVING Examples")
    print("=" * 60)

    # Create client
    client = GoVecClient(base_url="http://localhost:8000")

    # Check server health
    print("\n🏥 Checking server health...")
    if not client.health_check():
        print("❌ Server is not healthy. Please start GoVec server:")
        print("   cd /Volumes/work/govec && task run")
        return 1

    print("✅ Server is healthy")

    # Run examples
    example_1_single_insert(client)
    example_2_hybrid_insert(client)
    example_3_batch_insert(client)
    example_4_validation_error(client)

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)

    # Clean up
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
