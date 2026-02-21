#!/usr/bin/env python3
"""
Example 2: RETRIEVING - Vector Search Examples

This script demonstrates how to search for similar vectors in GoVec:
- Basic similarity search
- Top-K retrieval
- Metadata filtering (if supported)
- Hybrid search with sparse vectors
- Result iteration and display

Usage:
    python examples/02_search_vectors.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from govec_client import GoVecClient, SparseVector
from govec_client.exceptions import GoVecError


def setup_test_data(client: GoVecClient):
    """Insert test vectors for search examples."""
    print("\n📥 Setting up test data...")
    print("-" * 60)

    test_vectors = [
        {
            "id": "search_vec_1",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"category": "A", "priority": 1}
        },
        {
            "id": "search_vec_2",
            "vector": [0.15, 0.25, 0.35, 0.45],
            "metadata": {"category": "A", "priority": 2}
        },
        {
            "id": "search_vec_3",
            "vector": [0.5, 0.6, 0.7, 0.8],
            "metadata": {"category": "B", "priority": 1}
        },
        {
            "id": "search_vec_4",
            "vector": [-0.1, -0.2, -0.3, -0.4],
            "metadata": {"category": "B", "priority": 3}
        },
        {
            "id": "search_vec_5",
            "vector": [0.2, 0.2, 0.2, 0.2],
            "metadata": {"category": "C", "priority": 2}
        },
    ]

    for vec in test_vectors:
        try:
            client.insert(vec["id"], vec["vector"], vec["metadata"])
            print(f"✅ Inserted: {vec['id']}")
        except GoVecError as e:
            print(f"⚠️  {vec['id']} - {e} (may already exist)")


def example_1_basic_search(client: GoVecClient):
    """Perform basic similarity search."""
    print("\n🔍 Example 1: Basic similarity search")
    print("-" * 60)

    try:
        # Search for vectors similar to [0.1, 0.2, 0.3, 0.4]
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = client.search(vector=query_vector, k=5)

        print(f"Query vector: {query_vector}")
        print(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] ID: {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Metadata: {result.metadata}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_2_top_k_search(client: GoVecClient):
    """Retrieve top-K most similar vectors."""
    print("\n🔍 Example 2: Top-K search (k=3)")
    print("-" * 60)

    try:
        query_vector = [0.5, 0.6, 0.7, 0.8]
        results = client.search(vector=query_vector, k=3)

        print(f"Query vector: {query_vector}")
        print(f"Top {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}: {result.score:.4f}")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_3_hybrid_search(client: GoVecClient):
    """Search using hybrid (dense + sparse) vectors."""
    print("\n🔍 Example 3: Hybrid search (dense + sparse)")
    print("-" * 60)

    try:
        # First, insert a vector with sparse component
        sparse = SparseVector(
            indices=[10, 25, 100],
            values=[0.8, 0.6, 0.4]
        )

        client.insert(
            id="hybrid_search_target",
            vector=[0.3, 0.3, 0.3, 0.3],
            metadata={"type": "hybrid_target"},
            sparse_vector=sparse
        )
        print("✅ Inserted hybrid vector")

        # Search with matching sparse vector
        query_sparse = SparseVector(
            indices=[10, 25, 100, 200],
            values=[0.9, 0.7, 0.5, 0.3]
        )

        results = client.search(
            vector=[0.3, 0.3, 0.3, 0.3],
            k=5,
            sparse_vector=query_sparse
        )

        print(f"\nHybrid search results (dense + sparse matching):")
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}: {result.score:.4f}")
            if result.metadata:
                print(f"      Metadata: {result.metadata}")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_4_similarity_ranking(client: GoVecClient):
    """Demonstrate similarity score ranking."""
    print("\n🔍 Example 4: Similarity score ranking")
    print("-" * 60)

    try:
        # Query with a vector close to search_vec_1
        query_vector = [0.11, 0.21, 0.31, 0.41]  # Very similar to [0.1, 0.2, 0.3, 0.4]
        results = client.search(vector=query_vector, k=10)

        print(f"Query vector: {query_vector}")
        print(f"\nResults sorted by similarity (descending):\n")

        for i, result in enumerate(results, 1):
            bar_length = int(result.score * 50)  # Visual bar
            bar = "█" * bar_length
            print(f"  [{i}] {result.id:20s} {result.score:.4f} {bar}")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def main():
    """Run all search examples."""
    print("=" * 60)
    print("GoVec Python Client - RETRIEVING Examples")
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

    # Setup test data
    setup_test_data(client)

    # Run examples
    example_1_basic_search(client)
    example_2_top_k_search(client)
    example_3_hybrid_search(client)
    example_4_similarity_ranking(client)

    print("\n" + "=" * 60)
    print("✅ All search examples completed!")
    print("=" * 60)

    # Clean up
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
