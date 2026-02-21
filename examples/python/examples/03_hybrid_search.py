#!/usr/bin/env python3
"""
Example 3: Hybrid Search with Dense + Sparse Vectors

This script demonstrates advanced hybrid search combining:
- Dense vectors (semantic similarity)
- Sparse vectors (keyword/BM25 matching)

Use case: Semantic search with keyword boosting (e.g., "find similar documents
but prioritize exact keyword matches")

Usage:
    python examples/03_hybrid_search.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from govec_client import GoVecClient, SparseVector, VectorPayload
from govec_client.exceptions import GoVecError


def setup_hybrid_corpus(client: GoVecClient):
    """
    Insert a corpus of documents with both dense and sparse vectors.

    Simulates a real-world scenario where:
    - Dense vectors represent semantic embeddings (e.g., from BERT)
    - Sparse vectors represent BM25 scores for important keywords
    """
    print("\n📥 Setting up hybrid search corpus...")
    print("-" * 60)

    # Document 1: Python machine learning tutorial
    # Keywords: "python"=100, "machine"=200, "learning"=300, "tutorial"=400
    client.insert(
        id="doc_python_ml",
        vector=[0.8, 0.6, 0.4, 0.2],  # Dense semantic embedding
        sparse_vector=SparseVector(
            indices=[100, 200, 300, 400],
            values=[0.9, 0.8, 0.9, 0.5]
        ),
        metadata={
            "title": "Python Machine Learning Tutorial",
            "category": "AI"
        }
    )
    print("✅ Inserted: doc_python_ml")

    # Document 2: Deep learning with PyTorch
    # Keywords: "deep"=150, "learning"=300, "pytorch"=500
    client.insert(
        id="doc_pytorch_dl",
        vector=[0.75, 0.65, 0.45, 0.25],
        sparse_vector=SparseVector(
            indices=[150, 300, 500],
            values=[0.7, 0.9, 0.8]
        ),
        metadata={
            "title": "Deep Learning with PyTorch",
            "category": "AI"
        }
    )
    print("✅ Inserted: doc_pytorch_dl")

    # Document 3: Python web development
    # Keywords: "python"=100, "web"=600, "development"=700
    client.insert(
        id="doc_python_web",
        vector=[0.2, 0.3, 0.8, 0.7],
        sparse_vector=SparseVector(
            indices=[100, 600, 700],
            values=[0.9, 0.8, 0.7]
        ),
        metadata={
            "title": "Python Web Development Guide",
            "category": "Web"
        }
    )
    print("✅ Inserted: doc_python_web")

    # Document 4: Data science basics
    # Keywords: "data"=250, "science"=350, "basics"=450
    client.insert(
        id="doc_data_science",
        vector=[0.6, 0.5, 0.3, 0.1],
        sparse_vector=SparseVector(
            indices=[250, 350, 450],
            values=[0.85, 0.9, 0.6]
        ),
        metadata={
            "title": "Data Science Basics",
            "category": "AI"
        }
    )
    print("✅ Inserted: doc_data_science")


def example_1_dense_only_search(client: GoVecClient):
    """Search using only dense vectors (pure semantic similarity)."""
    print("\n🔍 Example 1: Dense-only search (semantic similarity)")
    print("-" * 60)

    try:
        # Query: "machine learning tutorial" embedding
        query_dense = [0.8, 0.6, 0.4, 0.2]

        results = client.search(vector=query_dense, k=4)

        print("Query: Semantic embedding for 'machine learning tutorial'")
        print(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Title: {result.metadata.get('title', 'N/A')}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_2_hybrid_search_ml(client: GoVecClient):
    """Hybrid search: semantic + keyword matching for 'machine learning'."""
    print("\n🔍 Example 2: Hybrid search - 'machine learning'")
    print("-" * 60)

    try:
        # Query with both dense and sparse components
        # Dense: semantic embedding for "machine learning"
        query_dense = [0.8, 0.6, 0.4, 0.2]

        # Sparse: keyword matches for "machine"=200, "learning"=300
        query_sparse = SparseVector(
            indices=[200, 300],  # "machine", "learning"
            values=[0.85, 0.95]
        )

        results = client.search(
            vector=query_dense,
            k=4,
            sparse_vector=query_sparse
        )

        print("Query: 'machine learning' (hybrid: semantic + keywords)")
        print(f"  Dense vector: {query_dense}")
        print(f"  Sparse keywords: machine (200), learning (300)")
        print(f"\nResults (ranked by combined score):\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Title: {result.metadata.get('title', 'N/A')}")
            print(f"      Category: {result.metadata.get('category', 'N/A')}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_3_hybrid_search_python(client: GoVecClient):
    """Hybrid search: boost results containing 'python' keyword."""
    print("\n🔍 Example 3: Hybrid search - 'python' keyword boost")
    print("-" * 60)

    try:
        # Query: general programming embedding + strong "python" keyword match
        query_dense = [0.5, 0.5, 0.5, 0.5]  # Neutral semantic embedding

        # Sparse: strong match on "python"=100
        query_sparse = SparseVector(
            indices=[100],  # "python"
            values=[1.0]    # Maximum weight
        )

        results = client.search(
            vector=query_dense,
            k=4,
            sparse_vector=query_sparse
        )

        print("Query: General programming query with 'python' keyword boost")
        print(f"  Sparse boost: python (100) = 1.0")
        print(f"\nResults (should prioritize python docs):\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Title: {result.metadata.get('title', 'N/A')}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def example_4_sparse_only_search(client: GoVecClient):
    """Search using only sparse vectors (pure keyword matching)."""
    print("\n🔍 Example 4: Sparse-only search (keyword matching)")
    print("-" * 60)

    try:
        # Query with neutral dense vector but strong sparse signal
        query_dense = [0.0, 0.0, 0.0, 0.0]  # Minimal semantic signal

        # Sparse: exact match on "learning"=300
        query_sparse = SparseVector(
            indices=[300],  # "learning"
            values=[1.0]
        )

        results = client.search(
            vector=query_dense,
            k=4,
            sparse_vector=query_sparse
        )

        print("Query: Keyword-only search for 'learning'")
        print(f"  Sparse keyword: learning (300) = 1.0")
        print(f"\nResults (ranked by keyword match):\n")

        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Title: {result.metadata.get('title', 'N/A')}")
            print()

    except GoVecError as e:
        print(f"❌ Error: {e}")


def main():
    """Run all hybrid search examples."""
    print("=" * 60)
    print("GoVec Python Client - Hybrid Search Examples")
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

    # Setup hybrid corpus
    try:
        setup_hybrid_corpus(client)
    except GoVecError as e:
        print(f"⚠️  Setup warning: {e} (documents may already exist)")

    # Run examples
    example_1_dense_only_search(client)
    example_2_hybrid_search_ml(client)
    example_3_hybrid_search_python(client)
    example_4_sparse_only_search(client)

    print("\n" + "=" * 60)
    print("✅ All hybrid search examples completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Dense vectors capture semantic similarity")
    print("  • Sparse vectors enable keyword/BM25 matching")
    print("  • Hybrid search combines both for better relevance")
    print("  • Token IDs in sparse vectors must be sorted ascending")
    print("=" * 60)

    # Clean up
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
