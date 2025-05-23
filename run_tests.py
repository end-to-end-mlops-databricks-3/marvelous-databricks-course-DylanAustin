"""Test runner script for hotel reservations preprocessing module.

This script provides different options for running tests with various configurations.
"""

import argparse
import importlib.util
import subprocess
import sys


def check_dependencies() -> bool:
    """Check if required testing dependencies are installed."""
    try:
        importlib.util.find_spec("pytest")

        return True
    except ImportError:
        print("❌ pytest is not installed!")
        print("Please install testing dependencies:")
        print("  pip install -r requirements-test.txt")
        print("  or")
        print("  pip install pytest")
        return False


def run_unit_tests(verbose: bool = True, coverage: bool = True) -> int:
    """Run unit tests only.

    :param verbose: Enable verbose output
    :param coverage: Generate coverage report
    :return: Exit code from pytest
    """
    if not check_dependencies():
        return 1

    cmd = ["python", "-m", "pytest", "tests/test_data_preprocessor.py"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=hotel_reservations", "--cov-report=html", "--cov-report=term"])

    return subprocess.call(cmd)


def run_integration_tests(verbose: bool = True) -> int:
    """Run integration tests.

    :param verbose: Enable verbose output
    :return: Exit code from pytest
    """
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_data_preprocessor.py::TestDataProcessor::test_integration_full_pipeline",
    ]

    if verbose:
        cmd.append("-v")

    return subprocess.call(cmd)


def run_all_tests(verbose: bool = True, coverage: bool = True) -> int:
    """Run all tests.

    :param verbose: Enable verbose output
    :param coverage: Generate coverage report
    :return: Exit code from pytest
    """
    cmd = ["python", "-m", "pytest", "tests/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=hotel_reservations", "--cov-report=html", "--cov-report=term"])

    return subprocess.call(cmd)


def run_specific_test(test_name: str, verbose: bool = True) -> int:
    """Run a specific test.

    :param test_name: Name of the test to run
    :param verbose: Enable verbose output
    :return: Exit code from pytest
    """
    cmd = ["python", "-m", "pytest", f"tests/test_data_preprocessor.py::{test_name}"]

    if verbose:
        cmd.append("-v")

    return subprocess.call(cmd)


def check_code_quality() -> int:
    """Run code quality checks.

    :return: Combined exit code from all quality checks
    """
    print("Running code quality checks...")

    # Run black formatting check
    print("\n1. Checking code formatting with black...")
    black_result = subprocess.call(["python", "-m", "black", "--check", "--diff", "hotel_reservations/", "tests/"])

    # Run isort import sorting check
    print("\n2. Checking import sorting with isort...")
    isort_result = subprocess.call(["python", "-m", "isort", "--check-only", "--diff", "hotel_reservations/", "tests/"])

    # Run flake8 linting
    print("\n3. Running flake8 linting...")
    flake8_result = subprocess.call(["python", "-m", "flake8", "hotel_reservations/", "tests/"])

    # Run mypy type checking
    print("\n4. Running mypy type checking...")
    mypy_result = subprocess.call(["python", "-m", "mypy", "hotel_reservations/"])

    return black_result + isort_result + flake8_result + mypy_result


def main() -> None:
    """Handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test runner for hotel reservations preprocessing module")
    parser.add_argument(
        "--type", choices=["unit", "integration", "all", "quality"], default="all", help="Type of tests to run"
    )
    parser.add_argument("--test", type=str, help="Run a specific test by name")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    verbose = not args.quiet
    coverage = not args.no_coverage

    if args.test:
        exit_code = run_specific_test(args.test, verbose)
    elif args.type == "unit":
        exit_code = run_unit_tests(verbose, coverage)
    elif args.type == "integration":
        exit_code = run_integration_tests(verbose)
    elif args.type == "quality":
        exit_code = check_code_quality()
    else:  # all
        print("Running all tests...")
        exit_code = run_all_tests(verbose, coverage)

        if exit_code == 0:
            print("\nRunning code quality checks...")
            quality_code = check_code_quality()
            exit_code = max(exit_code, quality_code)

    if exit_code == 0:
        print("\n✅ All checks passed!")
    else:
        print("\n❌ Some checks failed!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
