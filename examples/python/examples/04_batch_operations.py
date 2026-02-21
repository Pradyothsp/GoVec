#!/usr/bin/env python3
"""
Example 4: Batch Operations for Efficient Vector Management

This script demonstrates efficient batch operations:
- Batch insertion of multiple vectors
- Sequential vs batch performance comparison
- Large-scale data ingestion patterns

Usage:
    python examples/04_batch_operations.py
"""

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from govec_client import GoVecClient, VectorPayload
from govec_client.exceptions import GoVecError


def example_1_batch_insert(client: GoVecClient):
    """Demonstrate batch insertion."""
    print("\n📥 Example 1: Batch insertion")
    print("-" * 60)

    try:
        # Prepare batch of 10 vectors
        batch_size = 10
        vectors = [
            VectorPayload(
                id=f"batch_demo_{i}",
                vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                metadata={
                    "batch": "demo",
                    "index": i,
                    "squared": i * i
                }
            )
            for i in range(1, batch_size + 1)
        ]

        print(f"Inserting {batch_size} vectors in batch...")
        start_time = time.time()

        responses = client.insert_batch(vectors)

        elapsed = time.time() - start_time

        print(f"✅ Inserted {len(responses)} vectors in {elapsed:.3f}s")
        print(f"   Average: {elapsed/len(responses)*1000:.2f}ms per vector")

        # Show sample results
        print("\nSample results:")
        for i in range(min(3, len(responses))):
            print(f"  [{i+1}] {vectors[i].id} - Status: {responses[i].status}")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_2_large_batch(client: GoVecClient):
    """Insert larger batch (50 vectors)."""
    print("\n📥 Example 2: Large batch insertion (50 vectors)")
    print("-" * 60)

    try:
        batch_size = 50
        vectors = [
            VectorPayload(
                id=f"large_batch_{i}",
                vector=[
                    (i % 10) * 0.1,
                    (i % 10) * 0.2,
                    (i % 10) * 0.3,
                    (i % 10) * 0.4
                ],
                metadata={"batch": "large", "index": i}
            )
            for i in range(1, batch_size + 1)
        ]

        print(f"Inserting {batch_size} vectors...")
        start_time = time.time()

        responses = client.insert_batch(vectors)

        elapsed = time.time() - start_time

        print(f"✅ Completed in {elapsed:.3f}s")
        print(f"   Throughput: {len(responses)/elapsed:.1f} vectors/second")
        print(f"   Average latency: {elapsed/len(responses)*1000:.2f}ms per vector")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_3_batch_with_metadata(client: GoVecClient):
    """Batch insert with rich metadata."""
    print("\n📥 Example 3: Batch insert with rich metadata")
    print("-" * 60)

    try:
        # Simulate document embeddings with metadata
        documents = [
            {
                "id": "doc_001",
                "vector": [0.8, 0.6, 0.4, 0.2],
                "metadata": {
                    "title": "Introduction to Machine Learning",
                    "author": "Jane Doe",
                    "year": 2023,
                    "tags": ["AI", "ML", "tutorial"]
                }
            },
            {
                "id": "doc_002",
                "vector": [0.7, 0.5, 0.3, 0.1],
                "metadata": {
                    "title": "Deep Learning Fundamentals",
                    "author": "John Smith",
                    "year": 2024,
                    "tags": ["AI", "DL", "neural networks"]
                }
            },
            {
                "id": "doc_003",
                "vector": [0.6, 0.4, 0.2, 0.0],
                "metadata": {
                    "title": "Python Programming Guide",
                    "author": "Alice Johnson",
                    "year": 2023,
                    "tags": ["Python", "programming", "tutorial"]
                }
            },
        ]

        vectors = [
            VectorPayload(
                id=doc["id"],
                vector=doc["vector"],
                metadata=doc["metadata"]
            )
            for doc in documents
        ]

        responses = client.insert_batch(vectors)

        print(f"✅ Inserted {len(responses)} documents with metadata:")
        for i, doc in enumerate(documents):
            print(f"  [{i+1}] {doc['id']}")
            print(f"      Title: {doc['metadata']['title']}")
            print(f"      Tags: {', '.join(doc['metadata']['tags'])}")

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_4_batch_search(client: GoVecClient):
    """Perform multiple searches efficiently."""
    print("\n🔍 Example 4: Batch search operations")
    print("-" * 60)

    try:
        # Multiple query vectors
        queries = [
            {"name": "Query 1", "vector": [0.8, 0.6, 0.4, 0.2]},
            {"name": "Query 2", "vector": [0.1, 0.2, 0.3, 0.4]},
            {"name": "Query 3", "vector": [0.5, 0.5, 0.5, 0.5]},
        ]

        print(f"Executing {len(queries)} searches...")
        start_time = time.time()

        all_results = []
        for query in queries:
            results = client.search(vector=query["vector"], k=3)
            all_results.append((query["name"], results))

        elapsed = time.time() - start_time

        print(f"✅ Completed in {elapsed:.3f}s")
        print(f"   Average: {elapsed/len(queries)*1000:.2f}ms per search\n")

        # Show results
        for query_name, results in all_results:
            print(f"{query_name}:")
            for i, result in enumerate(results[:3], 1):
                print(f"  [{i}] {result.id}: {result.score:.4f}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def main():
    """Run all batch operation examples."""
    print("=" * 60)
    print("GoVec Python Client - Batch Operations")
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
    example_1_batch_insert(client)
    example_2_large_batch(client)
    example_3_batch_with_metadata(client)
    example_4_batch_search(client)

    print("\n" + "=" * 60)
    print("✅ All batch operation examples completed!")
    print("=" * 60)
    print("\nPerformance Tips:")
    print("  • Use batch operations for bulk data ingestion")
    print("  • Monitor throughput (vectors/second)")
    print("  • Consider connection pooling for concurrent requests")
    print("  • Use metadata for efficient filtering")
    print("=" * 60)

    # Clean up
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
