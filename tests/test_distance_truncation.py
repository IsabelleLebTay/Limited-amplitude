"""
Tests for distance_truncation.py module.

To run these tests:
    pytest tests/test_distance_truncation.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from limited_amplitude.distance_truncation import (
    load_predicted_amplitudes,
    load_species_references,
    load_metadata,
    calculate_mean_amplitude,
    create_amplitude_dataframe,
    prepare_amplitude_thresholds,
    apply_distance_truncation,
    get_truncation_summary
)


class TestCalculateMeanAmplitude:
    """Test the calculate_mean_amplitude function."""

    def test_mean_with_both_mics(self):
        """Test mean calculation when both mics have values."""
        result = calculate_mean_amplitude(-20.0, -22.0, 'use_available')
        assert result == -21.0

    def test_mean_with_left_only(self):
        """Test using left mic when right is missing."""
        result = calculate_mean_amplitude(-20.0, np.nan, 'use_available')
        assert result == -20.0

    def test_mean_with_right_only(self):
        """Test using right mic when left is missing."""
        result = calculate_mean_amplitude(np.nan, -22.0, 'use_available')
        assert result == -22.0

    def test_mean_with_both_missing(self):
        """Test when both mics are missing."""
        result = calculate_mean_amplitude(np.nan, np.nan, 'use_available')
        assert pd.isna(result)

    def test_require_both_mode(self):
        """Test require_both mode returns NaN when one mic is missing."""
        result = calculate_mean_amplitude(-20.0, np.nan, 'require_both')
        assert pd.isna(result)

        result = calculate_mean_amplitude(-20.0, -22.0, 'require_both')
        assert result == -21.0


class TestLoadFunctions:
    """Test data loading functions."""

    @pytest.fixture
    def sample_predicted_amps_csv(self, tmp_path):
        """Create sample predicted amplitudes CSV."""
        data = {
            'target_spp': ['BCCH', 'BCCH', 'BCCH', 'WOTH', 'WOTH'],
            'distance': [50, 100, 150, 50, 150],
            'canopy': [0, 0, 0, 1, 1],
            'SM2': [0, 0, 0, 0, 0],
            'predicted': [-25, -30, -35, -20, -28]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "predicted_amps.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_species_ref_csv(self, tmp_path):
        """Create sample species reference CSV."""
        data = {
            'species': ['BCCH', 'WOTH', 'RBNU'],
            'reference': ['BCCH', 'WOTH', 'BCCH']
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "species_ref.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_metadata_csv(self, tmp_path):
        """Create sample metadata CSV."""
        data = {
            'location': ['LOC1', 'LOC2', 'LOC3'],
            'canopy': [0, 1, 0],
            'SM2': [0, 0, 1]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "metadata.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_predicted_amplitudes(self, sample_predicted_amps_csv):
        """Test loading predicted amplitudes."""
        df = load_predicted_amplitudes(sample_predicted_amps_csv)

        assert isinstance(df, pd.DataFrame)
        assert 'target_spp' in df.columns
        assert 'distance' in df.columns
        assert 'predicted' in df.columns
        assert len(df) == 5

    def test_load_species_references(self, sample_species_ref_csv):
        """Test loading species references."""
        ref_dict = load_species_references(sample_species_ref_csv)

        assert isinstance(ref_dict, dict)
        assert ref_dict['BCCH'] == 'BCCH'
        assert ref_dict['WOTH'] == 'WOTH'
        assert ref_dict['RBNU'] == 'BCCH'
        assert len(ref_dict) == 3

    def test_load_metadata(self, sample_metadata_csv):
        """Test loading metadata."""
        df = load_metadata(metadata_file=sample_metadata_csv)

        assert isinstance(df, pd.DataFrame)
        assert 'location' in df.columns
        assert 'canopy' in df.columns
        assert 'SM2' in df.columns
        assert len(df) == 3

    def test_load_metadata_separate_files(self, tmp_path):
        """Test loading metadata from separate canopy and SM2 files."""
        # Create canopy file
        canopy_data = pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'canopy': [0, 1]
        })
        canopy_path = tmp_path / "canopy.csv"
        canopy_data.to_csv(canopy_path, index=False)

        # Create SM2 file
        sm2_data = pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'SM2': [0, 1]
        })
        sm2_path = tmp_path / "sm2.csv"
        sm2_data.to_csv(sm2_path, index=False)

        df = load_metadata(canopy_file=canopy_path, sm2_file=sm2_path)

        assert len(df) == 2
        assert 'canopy' in df.columns
        assert 'SM2' in df.columns


class TestCreateAmplitudeDataframe:
    """Test the create_amplitude_dataframe function."""

    @pytest.fixture
    def sample_tags_for_amplitude(self, tmp_path):
        """Create sample tags CSV with amplitude data."""
        data = {
            'location': ['LOC1', 'LOC1', 'LOC2'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00', '2024-06-01 06:00'],
            'species_code': ['BCCH', 'BCCH', 'WOTH'],
            'left_freq_filter_tag_peak_level_dbfs': [-20.0, -22.0, -18.0],
            'right_freq_filter_tag_peak_level_dbfs': [-21.0, -23.0, -19.0],
            'is_complete': [True, True, True],
            'vocalization': ['Song', 'Song', 'Song'],
            'aru_task_status': ['Transcribed', 'Transcribed', 'Transcribed']
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "tags_amp.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_recordings(self, tmp_path):
        """Create sample recordings CSV."""
        data = {
            'location': ['LOC1', 'LOC2', 'LOC3'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00', '2024-06-01 06:00']
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "recordings.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_metadata_df(self):
        """Create sample metadata dataframe."""
        return pd.DataFrame({
            'location': ['LOC1', 'LOC2', 'LOC3'],
            'canopy': [0, 1, 0],
            'SM2': [0, 0, 1]
        })

    def test_create_amplitude_dataframe_basic(self, sample_tags_for_amplitude,
                                              sample_recordings, sample_metadata_df):
        """Test basic amplitude dataframe creation."""
        result = create_amplitude_dataframe(
            tags_csv_path=sample_tags_for_amplitude,
            recordings_csv_path=sample_recordings,
            metadata_df=sample_metadata_df
        )

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'location' in result.columns
        assert 'recording_date_time' in result.columns

        # Should have species columns
        assert 'BCCH' in result.columns
        assert 'WOTH' in result.columns

        # Should have 3 recordings (including LOC3 with no detections)
        assert len(result) == 3

        # Check amplitude values (should be means)
        loc1_row = result[result['location'] == 'LOC1']
        # Mean of two BCCH detections: (-20.5 + -22.5) / 2 = -21.5
        assert abs(loc1_row['BCCH'].values[0] - (-21.5)) < 0.01

    def test_amplitude_dataframe_fills_gaps(self, sample_tags_for_amplitude,
                                           sample_recordings, sample_metadata_df):
        """Test that recordings without detections are included."""
        result = create_amplitude_dataframe(
            tags_csv_path=sample_tags_for_amplitude,
            recordings_csv_path=sample_recordings,
            metadata_df=sample_metadata_df
        )

        # LOC3 should be present but with NaN for species
        loc3_row = result[result['location'] == 'LOC3']
        assert len(loc3_row) == 1
        assert pd.isna(loc3_row['BCCH'].values[0])


class TestPrepareAmplitudeThresholds:
    """Test the prepare_amplitude_thresholds function."""

    @pytest.fixture
    def sample_predicted_amps(self):
        """Create sample predicted amplitudes dataframe."""
        data = {
            'target_spp': ['BCCH', 'BCCH', 'BCCH', 'BCCH'],
            'distance': [100, 150, 100, 150],
            'canopy': [0, 0, 1, 1],
            'SM2': [0, 0, 0, 0],
            'predicted': [-30, -35, -28, -33]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_species_ref(self):
        """Create sample species reference dict."""
        return {'BCCH': 'BCCH', 'RBNU': 'BCCH'}

    def test_prepare_thresholds(self, sample_predicted_amps, sample_species_ref):
        """Test threshold preparation."""
        thresholds = prepare_amplitude_thresholds(
            predicted_amps=sample_predicted_amps,
            species_references=sample_species_ref,
            distance_threshold=150.0
        )

        # Should have thresholds for both species
        assert 'BCCH' in thresholds
        assert 'RBNU' in thresholds

        # Check structure
        assert 0 in thresholds['BCCH']  # canopy type 0
        assert 1 in thresholds['BCCH']  # canopy type 1
        assert 'SM2_0' in thresholds['BCCH'][0]

        # Check values
        assert thresholds['BCCH'][0]['SM2_0'] == -35  # closed canopy at 150m
        assert thresholds['BCCH'][1]['SM2_0'] == -33  # open canopy at 150m


class TestApplyDistanceTruncation:
    """Test the apply_distance_truncation function."""

    @pytest.fixture
    def sample_amplitude_df(self):
        """Create sample amplitude dataframe."""
        return pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00'],
            'BCCH': [-32.0, -37.0],  # LOC1 loud (closer to 0), LOC2 quiet (more negative)
            'WOTH': [-25.0, -40.0]   # LOC1 loud, LOC2 quiet
        })

    @pytest.fixture
    def sample_metadata_for_truncation(self):
        """Create sample metadata."""
        return pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'canopy': [0, 0],
            'SM2': [0, 0]
        })

    @pytest.fixture
    def sample_predicted_for_truncation(self):
        """Create sample predicted amplitudes."""
        data = {
            'target_spp': ['BCCH', 'BCCH', 'WOTH', 'WOTH'],
            'distance': [150, 150, 150, 150],
            'canopy': [0, 1, 0, 1],
            'SM2': [0, 0, 0, 0],
            'predicted': [-35, -33, -30, -28]
        }
        return pd.DataFrame(data)

    def test_apply_truncation_basic(self, sample_amplitude_df,
                                    sample_predicted_for_truncation,
                                    sample_metadata_for_truncation):
        """Test basic truncation."""
        species_ref = {'BCCH': 'BCCH', 'WOTH': 'WOTH'}

        result = apply_distance_truncation(
            amplitude_df=sample_amplitude_df,
            predicted_amps=sample_predicted_for_truncation,
            species_references=species_ref,
            metadata_df=sample_metadata_for_truncation,
            distance_threshold=150.0,
            output_format='amplitude'
        )

        # LOC1 BCCH: -37 > -35 (threshold), should be kept
        assert pd.notna(result[result['location'] == 'LOC1']['BCCH'].values[0])

        # LOC2 BCCH: -32 < -35 (threshold), should be removed
        assert pd.isna(result[result['location'] == 'LOC2']['BCCH'].values[0])

    def test_truncation_binary_format(self, sample_amplitude_df,
                                      sample_predicted_for_truncation,
                                      sample_metadata_for_truncation):
        """Test binary output format."""
        species_ref = {'BCCH': 'BCCH', 'WOTH': 'WOTH'}

        result = apply_distance_truncation(
            amplitude_df=sample_amplitude_df,
            predicted_amps=sample_predicted_for_truncation,
            species_references=species_ref,
            metadata_df=sample_metadata_for_truncation,
            distance_threshold=150.0,
            output_format='binary'
        )

        # Should have 0s and 1s only
        assert result['BCCH'].isin([0, 1]).all()
        assert result['WOTH'].isin([0, 1]).all()


class TestTruncationSummary:
    """Test the get_truncation_summary function."""

    def test_summary_calculation(self):
        """Test summary statistics calculation."""
        original = pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00'],
            'BCCH': [-37.0, -32.0],
            'WOTH': [-25.0, -40.0]
        })

        truncated = pd.DataFrame({
            'location': ['LOC1', 'LOC2'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00'],
            'BCCH': [-37.0, np.nan],  # One removed
            'WOTH': [-25.0, np.nan]   # One removed
        })

        summary = get_truncation_summary(original, truncated)

        assert summary['total_recordings'] == 2
        assert summary['total_species'] == 2
        assert summary['original_detections'] == 4
        assert summary['truncated_detections'] == 2
        assert summary['removed_detections'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
