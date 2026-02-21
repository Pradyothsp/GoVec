#!/usr/bin/env python3
"""
Verification script for GoVec Python client installation.

This script checks:
1. Python version compatibility
2. Required dependencies
3. GoVec client library imports
4. GoVec server connectivity

Usage:
    python verify_installation.py
"""

import sys
import os

def check_python_version():
    """Check Python version is 3.8 or higher."""
    print("=" * 60)
    print("1. Checking Python version...")
    print("=" * 60)

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible (>=3.8)")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor} is not compatible.")
        print("   Required: Python 3.8 or higher")
        return False


def check_dependencies():
    """Check required dependencies are installed."""
    print("\n" + "=" * 60)
    print("2. Checking dependencies...")
    print("=" * 60)

    dependencies = ["requests", "urllib3"]
    all_ok = True

    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {dep:15s} version {version}")
        except ImportError:
            print(f"❌ {dep:15s} NOT FOUND")
            all_ok = False

    if not all_ok:
        print("\n💡 Install missing dependencies:")
        print("   pip install -r requirements.txt")

    return all_ok


def check_govec_client():
    """Check GoVec client library imports."""
    print("\n" + "=" * 60)
    print("3. Checking GoVec client library...")
    print("=" * 60)

    try:
        from govec_client import (
            GoVecClient,
            GoVecConfig,
            VectorPayload,
            SearchResult,
            SparseVector,
            GoVecError,
        )
        print("✅ GoVecClient imported successfully")
        print("✅ GoVecConfig imported successfully")
        print("✅ VectorPayload imported successfully")
        print("✅ SearchResult imported successfully")
        print("✅ SparseVector imported successfully")
        print("✅ GoVecError imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\n💡 Make sure you're running from examples/python directory:")
        print("   cd examples/python && python verify_installation.py")
        return False


def check_server_connectivity():
    """Check if GoVec server is running and accessible."""
    print("\n" + "=" * 60)
    print("4. Checking GoVec server connectivity...")
    print("=" * 60)

    try:
        from govec_client import GoVecClient

        # Try to connect
        client = GoVecClient(base_url="http://localhost:8000", timeout=5)

        print("Connecting to http://localhost:8000...")
        if client.health_check():
            print("✅ Server is healthy and responding")
            client.close()
            return True
        else:
            print("❌ Server is not responding")
            print("\n💡 Start the GoVec server:")
            print("   cd /Volumes/work/govec && task run")
            client.close()
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Start the GoVec server:")
        print("   cd /Volumes/work/govec && task run")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  GoVec Python Client - Installation Verification".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")

    # Run checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GoVec Client Library", check_govec_client),
        ("Server Connectivity", check_server_connectivity),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:25s} {status}")

    print("=" * 60)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to use the GoVec Python client.")
        print("\n📖 Next steps:")
        print("   • Read the README: cat README.md")
        print("   • Run examples: python examples/01_insert_vectors.py")
        print("   • Explore the API: python examples/02_search_vectors.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
