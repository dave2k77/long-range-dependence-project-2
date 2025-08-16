#!/usr/bin/env python3
"""
Test Runner for Long-Range Dependence Benchmarking Framework

This script runs the complete test suite with proper configuration
and provides a summary of results.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run the test suite."""
    print("🧪 RUNNING TEST SUITE")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pytest.ini").exists():
        print("❌ Error: pytest.ini not found. Please run from project root.")
        return False
    
    # Check if src directory exists
    if not Path("src").exists():
        print("❌ Error: src directory not found. Please run from project root.")
        return False
    
    # Check if tests directory exists
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Please run from project root.")
        return False
    
    print("✅ Project structure verified")
    print("📁 Current directory:", os.getcwd())
    print()
    
    # Run tests with coverage
    try:
        print("🚀 Starting test execution...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ All tests passed successfully!")
            return True
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        return False


def run_specific_tests(test_pattern=None):
    """Run specific tests based on pattern."""
    if test_pattern is None:
        return run_tests()
    
    print(f"🎯 Running tests matching pattern: {test_pattern}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"tests/test_{test_pattern}.py",
            "-v",
            "--tb=short"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Selected tests passed successfully!")
            return True
        else:
            print(f"\n❌ Some selected tests failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {str(e)}")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LRD Framework Tests")
    parser.add_argument(
        "--pattern", "-p",
        help="Run tests matching specific pattern (e.g., 'dfa', 'base')"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run only quick tests (skip slow ones)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Quick test mode - skipping slow tests")
        os.environ["PYTEST_ADDOPTS"] = "-m 'not slow'"
    
    if args.pattern:
        success = run_specific_tests(args.pattern)
    else:
        success = run_tests()
    
    if success:
        print("\n🎉 Test execution completed successfully!")
        print("📊 Coverage report generated in htmlcov/ directory")
        sys.exit(0)
    else:
        print("\n💥 Test execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
