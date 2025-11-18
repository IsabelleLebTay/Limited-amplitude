"""
Integration tests using reference data.

These tests validate the entire workflow against known reference outputs.
To run these tests, you must first add reference data to test_data/inputs/
and test_data/reference_outputs/ directories.

To run these tests:
    pytest tests/test_integration.py -v

To skip these tests if reference data is not available:
    pytest tests/test_integration.py -v -m "not integration"
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from limited_amplitude.occurrence import create_occurrence_dataframe
from limited_amplitude.distance_truncation import (
    load_predicted_amplitudes,
    load_species_references,
    load_metadata,
    create_amplitude_dataframe,
    apply_distance_truncation
)

# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"
INPUTS_DIR = TEST_DATA_DIR / "inputs"
REFERENCE_DIR = TEST_DATA_DIR / "reference_outputs"


@pytest.fixture
def check_test_data_available():
    """Check if test data is available."""
    required_input_files = [
        INPUTS_DIR / "tags.csv",
        INPUTS_DIR / "recordings.csv",
        INPUTS_DIR / "predicted_amplitudes.csv",
        INPUTS_DIR / "species_references.csv",
        INPUTS_DIR / "metadata.csv"
    ]

    required_reference_files = [
        REFERENCE_DIR / "amplitude_dataframe.csv",
        REFERENCE_DIR / "truncated_dataframe.csv"
    ]

    missing_inputs = [f for f in required_input_files if not f.exists()]
    missing_references = [f for f in required_reference_files if not f.exists()]

    if missing_inputs or missing_references:
        pytest.skip(
            f"Test data not available. Missing files:\n"
            f"Inputs: {missing_inputs}\n"
            f"References: {missing_references}\n"
            f"See tests/test_data/README.md for instructions."
        )


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for the full workflow."""

    def test_amplitude_dataframe_against_reference(self, check_test_data_available):
        """Test that amplitude dataframe matches reference output."""
        # Load inputs
        tags_path = INPUTS_DIR / "tags.csv"
        recordings_path = INPUTS_DIR / "recordings.csv"
        metadata_path = INPUTS_DIR / "metadata.csv"

        metadata_df = load_metadata(metadata_file=metadata_path)

        # Create amplitude dataframe
        result = create_amplitude_dataframe(
            tags_csv_path=tags_path,
            recordings_csv_path=recordings_path,
            metadata_df=metadata_df
        )

        # Load reference
        expected = pd.read_csv(REFERENCE_DIR / "amplitude_dataframe.csv")

        # Compare structure
        assert list(result.columns) == list(expected.columns), \
            "Column names don't match"

        assert len(result) == len(expected), \
            f"Different number of rows: {len(result)} vs {len(expected)}"

        # Compare values (allowing for small floating point differences)
        for col in result.columns:
            if col in ['location', 'recording_date_time']:
                # Exact match for identifiers
                pd.testing.assert_series_equal(
                    result[col].astype(str),
                    expected[col].astype(str),
                    check_names=False
                )
            else:
                # Approximate match for amplitude values
                pd.testing.assert_series_equal(
                    result[col],
                    expected[col],
                    check_names=False,
                    rtol=1e-5,
                    atol=1e-8
                )

    def test_truncated_dataframe_against_reference(self, check_test_data_available):
        """Test that truncated dataframe matches reference output."""
        # Load inputs
        tags_path = INPUTS_DIR / "tags.csv"
        recordings_path = INPUTS_DIR / "recordings.csv"
        metadata_path = INPUTS_DIR / "metadata.csv"
        predicted_amps_path = INPUTS_DIR / "predicted_amplitudes.csv"
        species_ref_path = INPUTS_DIR / "species_references.csv"

        metadata_df = load_metadata(metadata_file=metadata_path)
        predicted_amps = load_predicted_amplitudes(predicted_amps_path)
        species_references = load_species_references(species_ref_path)

        # Create amplitude dataframe
        amplitude_df = create_amplitude_dataframe(
            tags_csv_path=tags_path,
            recordings_csv_path=recordings_path,
            metadata_df=metadata_df
        )

        # Apply truncation (assuming 150m threshold as default)
        result = apply_distance_truncation(
            amplitude_df=amplitude_df,
            predicted_amps=predicted_amps,
            species_references=species_references,
            metadata_df=metadata_df,
            distance_threshold=150.0,
            output_format='amplitude'
        )

        # Load reference
        expected = pd.read_csv(REFERENCE_DIR / "truncated_dataframe.csv")

        # Compare structure
        assert list(result.columns) == list(expected.columns), \
            "Column names don't match"

        assert len(result) == len(expected), \
            f"Different number of rows: {len(result)} vs {len(expected)}"

        # Compare values
        for col in result.columns:
            if col in ['location', 'recording_date_time']:
                # Exact match for identifiers
                pd.testing.assert_series_equal(
                    result[col].astype(str),
                    expected[col].astype(str),
                    check_names=False
                )
            else:
                # Approximate match for amplitude values
                pd.testing.assert_series_equal(
                    result[col],
                    expected[col],
                    check_names=False,
                    rtol=1e-5,
                    atol=1e-8
                )

    def test_workflow_detection_counts(self, check_test_data_available):
        """Test that detection counts are reasonable."""
        # Load inputs
        tags_path = INPUTS_DIR / "tags.csv"
        recordings_path = INPUTS_DIR / "recordings.csv"
        metadata_path = INPUTS_DIR / "metadata.csv"
        predicted_amps_path = INPUTS_DIR / "predicted_amplitudes.csv"
        species_ref_path = INPUTS_DIR / "species_references.csv"

        metadata_df = load_metadata(metadata_file=metadata_path)
        predicted_amps = load_predicted_amplitudes(predicted_amps_path)
        species_references = load_species_references(species_ref_path)

        # Create amplitude dataframe
        amplitude_df = create_amplitude_dataframe(
            tags_csv_path=tags_path,
            recordings_csv_path=recordings_path,
            metadata_df=metadata_df
        )

        # Apply truncation
        truncated_df = apply_distance_truncation(
            amplitude_df=amplitude_df,
            predicted_amps=predicted_amps,
            species_references=species_references,
            metadata_df=metadata_df,
            distance_threshold=150.0
        )

        # Sanity checks
        species_cols = [col for col in amplitude_df.columns
                       if col not in ['location', 'recording_date_time']]

        original_count = amplitude_df[species_cols].notna().sum().sum()
        truncated_count = truncated_df[species_cols].notna().sum().sum()

        # Truncated should have fewer or equal detections
        assert truncated_count <= original_count, \
            "Truncated dataframe has more detections than original"

        # Should have removed at least some detections (unless threshold is very permissive)
        # This is dataset-dependent, so we just check the relationship
        print(f"Original detections: {original_count}")
        print(f"Truncated detections: {truncated_count}")
        print(f"Removed: {original_count - truncated_count} ({100*(original_count-truncated_count)/original_count:.1f}%)")


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency and validation."""

    def test_input_data_validity(self, check_test_data_available):
        """Validate that input data has correct structure."""
        # Check tags CSV
        tags = pd.read_csv(INPUTS_DIR / "tags.csv")
        required_cols = ['location', 'recording_date_time', 'species_code',
                        'left_freq_filter_tag_peak_level_dbfs',
                        'right_freq_filter_tag_peak_level_dbfs']
        for col in required_cols:
            assert col in tags.columns, f"Missing column in tags.csv: {col}"

        # Check recordings CSV
        recordings = pd.read_csv(INPUTS_DIR / "recordings.csv")
        assert 'location' in recordings.columns
        assert 'recording_date_time' in recordings.columns

        # Check metadata CSV
        metadata = pd.read_csv(INPUTS_DIR / "metadata.csv")
        assert 'location' in metadata.columns
        assert 'canopy' in metadata.columns
        assert 'SM2' in metadata.columns

        # Check predicted amplitudes CSV
        pred_amps = pd.read_csv(INPUTS_DIR / "predicted_amplitudes.csv")
        assert 'target_spp' in pred_amps.columns
        assert 'distance' in pred_amps.columns
        assert 'predicted' in pred_amps.columns

        # Check species references CSV
        spp_ref = pd.read_csv(INPUTS_DIR / "species_references.csv")
        assert 'species' in spp_ref.columns
        assert 'reference' in spp_ref.columns

    def test_reference_data_validity(self, check_test_data_available):
        """Validate that reference output data has correct structure."""
        # Check amplitude dataframe
        amp_df = pd.read_csv(REFERENCE_DIR / "amplitude_dataframe.csv")
        assert 'location' in amp_df.columns
        assert 'recording_date_time' in amp_df.columns

        # Should have species columns (4-letter codes)
        species_cols = [col for col in amp_df.columns
                       if col not in ['location', 'recording_date_time']]
        assert len(species_cols) > 0, "No species columns found"

        # Check truncated dataframe
        trunc_df = pd.read_csv(REFERENCE_DIR / "truncated_dataframe.csv")
        assert 'location' in trunc_df.columns
        assert 'recording_date_time' in trunc_df.columns

        # Should have same columns as amplitude dataframe
        assert set(amp_df.columns) == set(trunc_df.columns), \
            "Amplitude and truncated dataframes have different columns"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
