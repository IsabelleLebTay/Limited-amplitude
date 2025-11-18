"""
Script to generate reference output data from your input files.

This script processes your real data through the package and saves the outputs
as reference data for testing.

Usage:
    python tests/generate_reference_data.py

Before running:
1. Place your input files in tests/test_data/inputs/
   - tags.csv
   - recordings.csv
   - predicted_amplitudes.csv
   - species_references.csv
   - metadata.csv

2. Run this script to generate reference outputs

3. The outputs will be saved to tests/test_data/reference_outputs/
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from limited_amplitude.distance_truncation import (
    load_predicted_amplitudes,
    load_species_references,
    load_metadata,
    create_amplitude_dataframe,
    apply_distance_truncation,
    save_dataframe,
    get_truncation_summary
)


def main():
    """Generate reference output data."""
    # Define paths
    test_data_dir = Path(__file__).parent / "test_data"
    inputs_dir = test_data_dir / "inputs"
    outputs_dir = test_data_dir / "reference_outputs"

    # Check that input files exist
    required_files = [
        "tags.csv",
        "recordings.csv",
        "predicted_amplitudes.csv",
        "species_references.csv",
        "metadata.csv"
    ]

    missing_files = []
    for filename in required_files:
        if not (inputs_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print("ERROR: Missing required input files:")
        for filename in missing_files:
            print(f"  - {inputs_dir / filename}")
        print("\nPlease add these files before running this script.")
        print("See tests/test_data/README.md for file format requirements.")
        return 1

    print("=" * 70)
    print("GENERATING REFERENCE DATA")
    print("=" * 70)

    # Load inputs
    print("\n[1/5] Loading metadata...")
    metadata_df = load_metadata(metadata_file=inputs_dir / "metadata.csv")

    print("\n[2/5] Loading predicted amplitudes...")
    predicted_amps = load_predicted_amplitudes(inputs_dir / "predicted_amplitudes.csv")

    print("\n[3/5] Loading species references...")
    species_references = load_species_references(inputs_dir / "species_references.csv")

    # Create Output 1: Amplitude dataframe
    print("\n[4/5] Creating amplitude dataframe (Output 1)...")
    amplitude_df = create_amplitude_dataframe(
        tags_csv_path=inputs_dir / "tags.csv",
        recordings_csv_path=inputs_dir / "recordings.csv",
        metadata_df=metadata_df
    )

    # Save Output 1
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output1_path = outputs_dir / "amplitude_dataframe.csv"
    save_dataframe(amplitude_df, output1_path, "amplitude dataframe (Output 1)")

    # Create Output 2: Truncated dataframe
    print("\n[5/5] Creating truncated dataframe (Output 2)...")
    truncated_df = apply_distance_truncation(
        amplitude_df=amplitude_df,
        predicted_amps=predicted_amps,
        species_references=species_references,
        metadata_df=metadata_df,
        distance_threshold=150.0,  # Default threshold
        output_format='amplitude'
    )

    # Save Output 2
    output2_path = outputs_dir / "truncated_dataframe.csv"
    save_dataframe(truncated_df, output2_path, "truncated dataframe (Output 2)")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = get_truncation_summary(amplitude_df, truncated_df)

    print("\nReference data generated successfully!")
    print(f"\nOutput files saved to: {outputs_dir}")
    print(f"  - {output1_path.name}")
    print(f"  - {output2_path.name}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the generated output files to ensure they look correct")
    print("2. Run integration tests to validate:")
    print("   pytest tests/test_integration.py -v")
    print("3. Run all tests:")
    print("   pytest tests/ -v")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
