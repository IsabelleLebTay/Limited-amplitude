# Limited Amplitude

A Python package to apply limited amplitude to acoustic data.

## Description

This package applies distance-based truncation to acoustic bird detection data using amplitude measurements. It enables users to filter detections by estimating the distance at which birds were vocalizing based on their song amplitude.

### How It Works

1. **Input Data:** The package takes WildTrax CSV exports (tags and recordings) along with metadata about recording locations (ARU type and canopy openness).

2. **Species Mapping:** Your species of interest are mapped to reference species that have amplitude-distance calibration data.

3. **Amplitude Calculation:** For each detection, amplitude is calculated as the mean of left and right microphone peak RMS dBFS values. If one microphone has issues, only the functional microphone is used.

4. **Distance Estimation:** Using the calibration data from reference species, the package estimates detection distances based on measured amplitudes, accounting for ARU type (SM2 vs SM4) and forest canopy openness (open vs closed).

5. **Truncation:** Detections are filtered based on a user-specified distance threshold (e.g., 150m), removing detections estimated to be beyond that distance.

### Outputs

The package can produce three types of outputs:

**1. Occurrence Matrix (1/0)**
- Wide-format dataframe: `location`, `recording_date_time`, and species columns
- Values are binary: 1 (detected) or 0 (not detected)
- Includes all recordings from recordings CSV (even those with no detections)

**2. Count Matrix**
- Same structure as occurrence matrix
- Values are counts: number of individual detections per species per recording
- Includes all recordings from recordings CSV

**3. Amplitude Dataframe** (optional intermediate output)
- Wide-format dataframe with mean amplitude (dBFS) values
- Used internally but can be exported for analysis

### Required Input Files

1. **Tags CSV** - WildTrax tags export containing detection data with amplitude measurements
2. **Recordings CSV** - WildTrax recordings export to fill gaps where no detections occurred
3. **Metadata CSV** - Location information including:
   - ARU type (SM2 or SM4)
   - Canopy openness (open or closed)
4. **Predicted Amplitudes CSV** - Calibration data showing expected amplitudes for reference species at various distances
5. **Species References CSV** - Mapping between your species and the reference species used for calibration

## Installation

Since this package is not on PyPI, install it directly from the repository:

```bash
# Install in development/editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Quick Example

```python
from limited_amplitude import occurrence, distance_truncation

# 1. Load metadata (location, canopy, SM2/SM4 ARU type)
metadata_df = distance_truncation.load_metadata(
    metadata_file="data/metadata.csv"
)

# 2. Apply distance truncation to individual tags
truncated_tags = distance_truncation.apply_distance_truncation(
    tags_csv_path="data/wildtrax_tags.csv",
    metadata_df=metadata_df,
    distance_threshold=150.0  # Distance in meters
)

# 3. Convert to occurrence format (1/0)
occurrence_df = distance_truncation.convert_to_occurrence(
    truncated_tags=truncated_tags,
    recordings_csv_path="data/wildtrax_recordings.csv"
)

# OR convert to count format
count_df = distance_truncation.convert_to_counts(
    truncated_tags=truncated_tags,
    recordings_csv_path="data/wildtrax_recordings.csv"
)

# Save results
occurrence_df.to_csv("output/occurrence_150m.csv", index=False)
count_df.to_csv("output/count_150m.csv", index=False)
```

### Occurrence Matrix Without Distance Truncation

You can also create simple occurrence matrices without distance truncation:

```python
from limited_amplitude import occurrence

# Create occurrence matrix (no distance filtering)
df_occ = occurrence.create_occurrence_dataframe(
    tags_csv_path="data/wildtrax_tags.csv",
    recordings_csv_path="data/wildtrax_recordings.csv",
    filter_vocalization='Song'  # Optional: filter by vocalization type
)
```

### Additional Examples

See `examples/example_workflow.ipynb` for a complete workflow including:
- Multiple distance thresholds
- Comparing truncated vs non-truncated data
- Custom filtering options

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd Limited-amplitude

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Project Structure

```
Limited-amplitude/
├── data/
│   ├── default_exclude_codes.txt       # Default species codes to exclude
│   ├── all_spp_predicted_amplitudes.csv # Amplitude-distance calibration data
│   └── species and references.csv       # Species to reference species mapping
├── src/
│   └── limited_amplitude/
│       ├── __init__.py            # Package initialization
│       ├── occurrence.py          # Create occurrence/detection matrices
│       └── distance_truncation.py # Amplitude-based distance truncation
├── examples/
│   └── example_workflow.ipynb     # Example notebook demonstrating workflow
├── tests/
│   ├── test_occurrence.py         # Unit tests for occurrence.py
│   ├── test_distance_truncation.py # Unit tests for distance_truncation.py
│   └── test_integration.py        # Integration tests
├── pyproject.toml                 # Project configuration
├── pytest.ini                     # Test configuration
└── README.md                      # This file
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Requirements

- Python 3.11 or higher
- pandas >= 2.0
- numpy >= 1.24

## Roadmap

- [ ] Command-line interface (currently in development)
- [ ] Additional output formats (e.g., spatial formats for GIS)

## License

MIT
