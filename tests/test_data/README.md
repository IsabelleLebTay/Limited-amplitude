# Test Data Directory

This directory contains test data for the `limited_amplitude` package.

## Directory Structure

```
test_data/
├── README.md                    # This file
├── reference_outputs/           # Expected output files for validation
│   ├── amplitude_dataframe.csv  # Expected Output 1: Mean amplitude dataframe
│   └── truncated_dataframe.csv  # Expected Output 2: Truncated amplitude dataframe
└── inputs/                      # Input files for testing (to be added)
    ├── tags.csv                 # WildTrax tags export
    ├── recordings.csv           # WildTrax recordings export
    ├── predicted_amplitudes.csv # Reference species amplitude predictions
    ├── species_references.csv   # Mapping of species to reference species
    └── metadata.csv             # Location metadata (canopy, SM2 status)
```

## Required Input Files

### 1. `tags.csv` - WildTrax Tags Export
Required columns:
- `location`: Location identifier
- `recording_date_time`: Recording timestamp
- `species_code`: 4-letter species code
- `left_freq_filter_tag_peak_level_dbfs`: Left mic amplitude
- `right_freq_filter_tag_peak_level_dbfs`: Right mic amplitude
- `is_complete`: Boolean indicating complete recording
- `vocalization`: Vocalization type (e.g., "Song")
- `aru_task_status`: Task status (e.g., "Transcribed")

### 2. `recordings.csv` - WildTrax Recordings Export
Required columns:
- `location`: Location identifier
- `recording_date_time`: Recording timestamp

This file ensures all recordings are included in outputs, even those with no detections.

### 3. `predicted_amplitudes.csv` - Reference Amplitudes
Required columns:
- `target_spp`: Reference species code
- `distance`: Distance in meters
- `canopy`: Canopy status (0=closed, 1=open)
- `SM2`: ARU type (0=SM4, 1=SM2)
- `predicted`: Predicted amplitude threshold at this distance

This file contains amplitude predictions for reference species at various distances,
accounting for canopy openness and ARU type.

### 4. `species_references.csv` - Species Reference Mapping
Required columns:
- `species`: Species code in your data
- `reference`: Reference species to use for amplitude thresholds

Maps each species in your dataset to a reference species that has amplitude data.

### 5. `metadata.csv` - Location Metadata
Required columns:
- `location`: Location identifier
- `canopy`: Canopy openness (0=closed canopy, 1=open canopy)
- `SM2`: ARU type (0=SM4, 1=SM2)

Provides metadata about each recording location.

## Expected Reference Outputs

### Output 1: `amplitude_dataframe.csv`
The first output is a wide-format dataframe where:
- First two columns: `location`, `recording_date_time`
- Remaining columns: Species codes (4-letter codes)
- Values: Mean amplitude (dBFS) for each species at each location-time

This represents all detections with their mean amplitude values.

### Output 2: `truncated_dataframe.csv`
The second output is the amplitude-truncated version where:
- Same structure as Output 1
- Values: Mean amplitude for detections within the distance threshold
- Detections beyond the threshold are set to NaN

## Adding Your Test Data

To add your reference data for testing:

1. **Create the `inputs/` directory**:
   ```bash
   mkdir test_data/inputs
   ```

2. **Copy your input files** to `test_data/inputs/`:
   - Export tags and recordings from WildTrax
   - Prepare your predicted amplitudes CSV
   - Create species reference mapping
   - Create location metadata CSV

3. **Run the package** on your test data to generate expected outputs

4. **Save the outputs** to `reference_outputs/`:
   ```python
   # After running your data through the package
   amplitude_df.to_csv('test_data/reference_outputs/amplitude_dataframe.csv', index=False)
   truncated_df.to_csv('test_data/reference_outputs/truncated_dataframe.csv', index=False)
   ```

5. **Run tests** to validate:
   ```bash
   pytest tests/ -v
   ```

## Using Reference Data in Tests

Tests will automatically compare outputs against the reference files:

```python
# Example test
def test_against_reference_data():
    # Process test inputs
    result = create_amplitude_dataframe(
        tags_csv_path='test_data/inputs/tags.csv',
        recordings_csv_path='test_data/inputs/recordings.csv',
        metadata_df=metadata
    )

    # Load reference output
    expected = pd.read_csv('test_data/reference_outputs/amplitude_dataframe.csv')

    # Compare
    pd.testing.assert_frame_equal(result, expected)
```

## Notes

- All CSV files should be UTF-8 encoded
- Date/time format should be consistent (recommend: YYYY-MM-DD HH:MM)
- Species codes should be 4-letter uppercase codes
- Amplitude values are typically negative (dBFS scale)
- Missing values should be represented as empty cells or NaN

## Questions?

If you have questions about test data format or requirements, please refer to:
- Package README: `../README.md`
- Module documentation: `../src/limited_amplitude/`
