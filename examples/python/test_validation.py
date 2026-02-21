#!/usr/bin/env python3
"""
Quick test script to verify validation error handling works correctly.

This script tests that ValidationErrors are properly raised and caught
for various invalid inputs.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from govec_client import GoVecClient, SparseVector
from govec_client.exceptions import ValidationError


def test_empty_vector():
    """Test that empty vector raises ValidationError."""
    print("Test 1: Empty vector")
    client = GoVecClient(base_url="http://localhost:8000")

    try:
        client.insert(id="test", vector=[])
        print("  ❌ FAILED: Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  ✅ PASSED: Caught ValidationError: {e}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_empty_search_vector():
    """Test that empty search vector raises ValidationError."""
    print("\nTest 2: Empty search vector")
    client = GoVecClient(base_url="http://localhost:8000")

    try:
        client.search(vector=[])
        print("  ❌ FAILED: Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  ✅ PASSED: Caught ValidationError: {e}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_invalid_k():
    """Test that k <= 0 raises ValidationError."""
    print("\nTest 3: Invalid k value (k=0)")
    client = GoVecClient(base_url="http://localhost:8000")

    try:
        client.search(vector=[0.1, 0.2, 0.3], k=0)
        print("  ❌ FAILED: Should have raised ValidationError")
        return False
    except ValidationError as e:
        print(f"  ✅ PASSED: Caught ValidationError: {e}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_invalid_sparse_vector():
    """Test that invalid sparse vector raises ValidationError."""
    print("\nTest 4: Invalid sparse vector (mismatched lengths)")
    client = GoVecClient(base_url="http://localhost:8000")

    try:
        # This should fail in SparseVector.__post_init__
        sparse = SparseVector(indices=[1, 2, 3], values=[0.5, 0.6])
        print("  ❌ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ PASSED: Caught ValueError: {e}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def test_unsorted_sparse_indices():
    """Test that unsorted sparse indices raise ValidationError."""
    print("\nTest 5: Unsorted sparse vector indices")
    client = GoVecClient(base_url="http://localhost:8000")

    try:
        # This should fail - indices not sorted
        sparse = SparseVector(indices=[3, 1, 2], values=[0.5, 0.6, 0.7])
        print("  ❌ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✅ PASSED: Caught ValueError: {e}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: Wrong exception type: {type(e).__name__}: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("GoVec Python Client - Validation Error Tests")
    print("=" * 60)
    print()

    tests = [
        test_empty_vector,
        test_empty_search_vector,
        test_invalid_k,
        test_invalid_sparse_vector,
        test_unsorted_sparse_indices,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All validation tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
