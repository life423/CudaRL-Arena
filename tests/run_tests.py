"""
Test runner for CudaRL-Arena with comprehensive options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_pytest_with_args(args: argparse.Namespace) -> int:
    """Run pytest with parsed arguments."""
    cmd = ["python", "-m", "pytest"]
    
    # Test categories
    if args.unit:
        cmd.extend(["tests/unit"])
    elif args.integration:
        cmd.extend(["tests/integration"])
    elif args.performance:
        cmd.extend(["tests/performance"])
    else:
        # Run all tests by default
        cmd.extend(["tests/unit", "tests/integration", "tests/performance"])
    
    # GPU tests
    if args.gpu_only:
        cmd.extend(["-m", "gpu"])
    elif args.no_gpu:
        cmd.extend(["-m", "not gpu"])
    
    # Performance settings
    if args.benchmark:
        cmd.extend(["-m", "performance", "--benchmark-only"])
    
    # Parallel execution
    if args.parallel and args.parallel > 1:
        cmd.extend(["-n", str(args.parallel)])
    
    # Coverage
    if args.coverage:
        cmd.extend(["--cov=python/cudarl", "--cov-report=html"])
    
    # Verbosity
    if args.verbose:
        cmd.extend(["-v"])
    elif args.quiet:
        cmd.extend(["-q"])
    
    # Output format
    if args.html:
        cmd.extend(["--html=tests/reports/report.html"])
    
    # Additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="CudaRL-Arena Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --unit             # Run only unit tests
  python tests/run_tests.py --gpu-only         # Run only GPU tests
  python tests/run_tests.py --benchmark        # Run performance tests
  python tests/run_tests.py --parallel 4       # Run with 4 workers
  python tests/run_tests.py --coverage         # Run with coverage
        """
    )
    
    # Test categories
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit", action="store_true",
        help="Run only unit tests"
    )
    test_group.add_argument(
        "--integration", action="store_true",
        help="Run only integration tests"
    )
    test_group.add_argument(
        "--performance", action="store_true",
        help="Run only performance tests"
    )
    
    # GPU settings
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu-only", action="store_true",
        help="Run only GPU-requiring tests"
    )
    gpu_group.add_argument(
        "--no-gpu", action="store_true",
        help="Skip GPU tests (CPU only)"
    )
    
    # Performance options
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark tests only"
    )
    
    # Execution options
    parser.add_argument(
        "--parallel", "-n", type=int, metavar="N",
        help="Run tests in parallel with N workers"
    )
    
    # Coverage
    parser.add_argument(
        "--coverage", action="store_true",
        help="Generate coverage report"
    )
    
    # Output options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    verbosity_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet output"
    )
    
    parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML report"
    )
    
    # Pass-through args
    parser.add_argument(
        "pytest_args", nargs="*",
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    # Create reports directory
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    return run_pytest_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
