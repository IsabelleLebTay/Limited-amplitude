"""
Command-line interface for limited-amplitude.
"""

import argparse
import sys
from . import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="limited-amplitude",
        description="A Python package for ecology analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
    )

    # Example subcommand - replace with your actual commands
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis on ecology data",
    )
    analyze_parser.add_argument(
        "input",
        help="Input data file",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file (optional)",
        default=None,
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "analyze":
        print(f"Analyzing {args.input}...")
        if args.output:
            print(f"Output will be saved to {args.output}")
        # Add your analysis logic here
        print("Analysis complete (placeholder)")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
