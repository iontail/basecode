#!/usr/bin/env python3
"""
Test runner script for the deep learning research codebase.
Provides easy interface to run different types of tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✅ {description or 'Command'} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure you have the required testing tools installed:")
        print("pip install pytest coverage")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for the deep learning research codebase"
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=['unit', 'integration', 'all'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Run tests with coverage report'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Run specific test file'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        help='Run specific test method'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    
    args = parser.parse_args()
    
    # Base command
    if args.coverage:
        base_cmd = ['python', '-m', 'coverage', 'run', '-m', 'pytest']
    else:
        base_cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        base_cmd.append('-v')
    
    # Add parallel execution
    if args.parallel:
        base_cmd.extend(['-n', 'auto'])
    
    # Add test selection
    if args.file:
        base_cmd.append(f"tests/{args.file}")
    elif args.method:
        base_cmd.extend(['-k', args.method])
    else:
        base_cmd.append('tests/')
    
    # Add markers based on test type
    if args.type == 'unit':
        base_cmd.extend(['-m', 'unit'])
    elif args.type == 'integration':
        base_cmd.extend(['-m', 'integration'])
    
    # Run tests
    success = run_command(base_cmd, f"Running {args.type} tests")
    
    # Run coverage report if requested
    if args.coverage and success:
        print("\n" + "="*60)
        print("Generating coverage report...")
        print("="*60)
        
        # Generate coverage report
        run_command(
            ['python', '-m', 'coverage', 'report', '-m'], 
            "Coverage report"
        )
        
        # Generate HTML coverage report
        run_command(
            ['python', '-m', 'coverage', 'html'], 
            "HTML coverage report"
        )
        print("\nHTML coverage report generated in 'htmlcov/' directory")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()