"""
Tests for occurrence.py module.

To run these tests:
    pytest tests/test_occurrence.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from limited_amplitude.occurrence import (
    load_exclude_codes,
    create_occurrence_dataframe,
    save_occurrence_dataframe,
    get_occurrence_summary
)


class TestLoadExcludeCodes:
    """Test the load_exclude_codes function."""

    def test_load_default_exclude_codes(self):
        """Test loading default exclude codes."""
        codes = load_exclude_codes()

        # Should return a list
        assert isinstance(codes, list)

        # Should contain expected codes
        expected_codes = ['NONE', 'RESQ', 'RTHA', 'UNWO', 'UNKN']
        for code in expected_codes:
            assert code in codes

        # Should not include comments or empty lines
        for code in codes:
            assert not code.startswith('#')
            assert len(code) > 0

    def test_load_custom_exclude_codes(self, tmp_path):
        """Test loading custom exclude codes from file."""
        # Create a temporary exclude codes file
        exclude_file = tmp_path / "custom_exclude.txt"
        exclude_file.write_text("# Custom exclude codes\nTEST\nDUMY\n\n# Another comment\nFAKE\n")

        codes = load_exclude_codes(exclude_file)

        assert codes == ['TEST', 'DUMY', 'FAKE']

    def test_missing_exclude_file(self, tmp_path):
        """Test behavior when exclude file doesn't exist."""
        missing_file = tmp_path / "nonexistent.txt"

        codes = load_exclude_codes(missing_file)

        # Should return empty list with warning
        assert codes == []


class TestCreateOccurrenceDataframe:
    """Test the create_occurrence_dataframe function."""

    @pytest.fixture
    def sample_tags_csv(self, tmp_path):
        """Create a sample tags CSV for testing."""
        data = {
            'location': ['LOC1', 'LOC1', 'LOC1', 'LOC2', 'LOC2'],
            'recording_date_time': ['2024-06-01 06:00', '2024-06-01 06:00', '2024-06-01 07:00',
                                   '2024-06-01 06:00', '2024-06-01 06:00'],
            'species_code': ['BCCH', 'BCCH', 'WOTH', 'BCCH', 'NONE'],
            'is_complete': [True, True, True, True, True],
            'vocalization': ['Song', 'Song', 'Song', 'Song', 'Song'],
            'latitude': [45.5, 45.5, 45.5, 45.6, 45.6],
            'longitude': [-75.5, -75.5, -75.5, -75.6, -75.6],
            'task_comments': ['', '', '', '', '']
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "tags.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_create_occurrence_basic(self, sample_tags_csv):
        """Test basic occurrence dataframe creation."""
        # Use exclude codes to remove NONE
        exclude_codes = ['NONE']

        result = create_occurrence_dataframe(
            tags_csv_path=sample_tags_csv,
            group_by=['location', 'recording_date_time'],
            exclude_codes=exclude_codes,
            filter_complete_only=True
        )

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert 'location' in result.columns
        assert 'recording_date_time' in result.columns

        # Check species columns
        assert 'BCCH' in result.columns
        assert 'WOTH' in result.columns
        assert 'NONE' not in result.columns  # Should be excluded

        # Check values are presence/absence (1 or 0)
        species_cols = ['BCCH', 'WOTH']
        for col in species_cols:
            assert result[col].isin([0, 1]).all()

    def test_occurrence_grouping(self, sample_tags_csv):
        """Test that grouping works correctly."""
        result = create_occurrence_dataframe(
            tags_csv_path=sample_tags_csv,
            group_by=['location', 'recording_date_time'],
            exclude_codes=['NONE']
        )

        # Should have 3 unique location-time combinations
        # LOC1 2024-06-01 06:00 (2 BCCH detections -> 1)
        # LOC1 2024-06-01 07:00 (1 WOTH detection -> 1)
        # LOC2 2024-06-01 06:00 (1 BCCH, 1 NONE -> BCCH=1)
        assert len(result) == 3

        # Check specific values
        loc1_0600 = result[(result['location'] == 'LOC1') &
                          (result['recording_date_time'] == '2024-06-01 06:00')]
        assert loc1_0600['BCCH'].values[0] == 1

        loc1_0700 = result[(result['location'] == 'LOC1') &
                          (result['recording_date_time'] == '2024-06-01 07:00')]
        assert loc1_0700['WOTH'].values[0] == 1

    def test_occurrence_summary(self, sample_tags_csv):
        """Test get_occurrence_summary function."""
        result = create_occurrence_dataframe(
            tags_csv_path=sample_tags_csv,
            exclude_codes=['NONE']
        )

        summary = get_occurrence_summary(result)

        assert 'total_records' in summary
        assert 'total_species' in summary
        assert 'species_list' in summary
        assert summary['total_records'] == len(result)
        assert 'BCCH' in summary['species_list']


class TestSaveOccurrenceDataframe:
    """Test the save_occurrence_dataframe function."""

    def test_save_creates_file(self, tmp_path):
        """Test that save function creates a file."""
        # Create a simple dataframe
        df = pd.DataFrame({
            'location': ['LOC1'],
            'recording_date_time': ['2024-06-01 06:00'],
            'BCCH': [1]
        })

        output_path = tmp_path / "output" / "occurrence.csv"

        save_occurrence_dataframe(df, output_path)

        # Check file exists
        assert output_path.exists()

        # Check content
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 1
        assert 'BCCH' in loaded.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
