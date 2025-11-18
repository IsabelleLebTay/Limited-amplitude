"""
Multi-species occurrence dataframe generation from WildTrax data.

This module processes WildTrax CSV exports to create species occurrence dataframes
with presence/absence or count data for each species at each location and recording time.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


def load_exclude_codes(exclude_file: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Load species codes to exclude from a text file.

    Args:
        exclude_file: Path to file containing species codes to exclude (one per line).
                     Lines starting with # are treated as comments.
                     If None, uses default exclude codes from package data folder.

    Returns:
        List of species codes to exclude
    """
    if exclude_file is None:
        # Use default exclude codes from package data folder
        package_dir = Path(__file__).parent.parent.parent
        exclude_file = package_dir / "data" / "default_exclude_codes.txt"
    else:
        exclude_file = Path(exclude_file)

    if not exclude_file.exists():
        print(f"Warning: Exclude codes file not found at {exclude_file}")
        return []

    exclude_codes = []
    with open(exclude_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                exclude_codes.append(line)

    return exclude_codes


def create_occurrence_dataframe(
    tags_csv_path: Union[str, Path],
    recordings_csv_path: Optional[Union[str, Path]] = None,
    group_by: List[str] = None,
    metadata_cols: List[str] = None,
    exclude_codes: Optional[List[str]] = None,
    exclude_file: Optional[Union[str, Path]] = None,
    filter_complete_only: bool = True,
    filter_vocalization: Optional[str] = None
) -> pd.DataFrame:
    """
    Create multi-species occurrence dataframe from WildTrax tags export.

    This function processes WildTrax data to create a wide-format dataframe where:
    - Index columns are location identifiers (e.g., location, recording_date_time)
    - Species columns contain presence/absence (1/0) data
    - All location + recording_date_time combinations from recordings CSV are included (if provided)

    Args:
        tags_csv_path: Path to WildTrax tags CSV export
        recordings_csv_path: Optional path to WildTrax recordings CSV. If provided, ensures all
                            location + recording_date_time combinations are included (with 0s for no detections)
        group_by: Columns to group by (default: ['location', 'recording_date_time'])
        metadata_cols: Additional metadata columns to include (default: None for clean species matrix)
        exclude_codes: List of species codes to exclude. If None, uses exclude_file.
        exclude_file: Path to file with species codes to exclude. If None, uses default.
        filter_complete_only: If True, only include rows where is_complete == True
        filter_vocalization: If specified, filter to only this vocalization type (e.g., 'Song')

    Returns:
        DataFrame with location/time columns and species presence/absence
    """
    # Set defaults
    if group_by is None:
        group_by = ['location', 'recording_date_time']

    if metadata_cols is None:
        metadata_cols = []

    # Load exclude codes
    if exclude_codes is None:
        exclude_codes = load_exclude_codes(exclude_file)

    print(f"Loading data from: {tags_csv_path}")
    data = pd.read_csv(tags_csv_path)
    print(f"Loaded {len(data)} rows")

    # Apply filters
    if filter_complete_only and 'is_complete' in data.columns:
        # Handle both boolean (True/False) and string ('t'/'f') values
        data = data[data['is_complete'].isin([True, 't', 'T', 'true', 'True'])]
        print(f"Filtered to {len(data)} complete rows")

    if filter_vocalization and 'vocalization' in data.columns:
        data = data[data['vocalization'] == filter_vocalization]
        print(
            f"Filtered to {len(data)} rows with vocalization = {filter_vocalization}")

    # Create dummy column for pivoting
    data['dummies'] = data['species_code']

    # Create one-hot encoded columns for species
    abundance = pd.get_dummies(
        data=data,
        columns=['dummies'],
        prefix='',
        prefix_sep=''
    )

    # Get list of all species in data
    full_species_list = list(data['species_code'].unique())

    # Filter out excluded species
    select_species = [s for s in full_species_list if s not in exclude_codes]
    print(
        f"Keeping {len(select_species)} species (excluded {len(full_species_list) - len(select_species)})")

    # Build column list: group_by columns + metadata + selected species
    columns_to_keep = group_by.copy()

    # Only include metadata columns that actually exist in the data
    existing_metadata_cols = []
    for col in metadata_cols:
        if col in abundance.columns and col not in columns_to_keep:
            columns_to_keep.append(col)
            existing_metadata_cols.append(col)

    columns_to_keep.extend(select_species)

    # Select only needed columns
    df_subset = abundance[columns_to_keep]

    # Create aggregation dictionary
    # For species columns: presence/absence (1 if any detection, 0 otherwise)
    species_agg = {species: lambda x: 1 if x.sum(
    ) > 0 else 0 for species in select_species}

    # For metadata columns: take first value (only for columns that actually exist)
    metadata_agg = {col: 'first' for col in existing_metadata_cols}

    # Combine aggregation dictionaries
    agg_dict = {**metadata_agg, **species_agg}

    # Group by specified columns
    grouped = df_subset.groupby(group_by).agg(agg_dict)

    # Reset index to make group_by columns regular columns
    grouped = grouped.reset_index()

    # If recordings CSV provided, merge to ensure all recordings are included
    if recordings_csv_path is not None:
        print(f"Loading recordings from {recordings_csv_path}...")
        recordings_df = pd.read_csv(recordings_csv_path)

        # Apply same filters to recordings as we did to tags
        if filter_complete_only and 'task_is_complete' in recordings_df.columns:
            # Handle both boolean (True/False) and string ('t'/'f') values
            recordings_df = recordings_df[recordings_df['task_is_complete'].isin(
                [True, 't', 'T', 'true', 'True'])]
            print(f"Filtered recordings to {len(recordings_df)} complete tasks")

        all_recordings = recordings_df[group_by].drop_duplicates()
        print(f"Found {len(all_recordings)} unique {'/'.join(group_by)} combinations in recordings")

        # Merge to include all recordings
        grouped = all_recordings.merge(
            grouped,
            on=group_by,
            how='left'
        )

        # Fill NaN with 0 for species columns (recordings with no detections)
        for species in select_species:
            if species in grouped.columns:
                grouped[species] = grouped[species].fillna(0).astype(int)

        print(f"Final occurrence dataframe includes all {len(grouped)} recordings")
    else:
        print(
            f"Created occurrence dataframe with {len(grouped)} unique {'/'.join(group_by)} combinations")

    return grouped


def get_occurrence_summary(occurrence_df: pd.DataFrame, group_by: List[str] = None) -> dict:
    """
    Get summary statistics of the occurrence dataframe.

    Args:
        occurrence_df: Occurrence dataframe to summarize
        group_by: Columns used for grouping (default: ['location', 'recording_date_time'])

    Returns:
        Dictionary with summary information
    """
    if group_by is None:
        group_by = ['location', 'recording_date_time']

    # Identify metadata columns (non-species, non-grouping columns)
    # Assume species codes are 4 characters
    metadata_cols = [col for col in occurrence_df.columns
                     if col not in group_by and len(col) != 4]

    species_cols = [col for col in occurrence_df.columns
                    if col not in group_by and col not in metadata_cols]

    summary = {
        'total_records': len(occurrence_df),
        'total_species': len(species_cols),
        'species_list': species_cols,
        'detections_per_species': {
            species: occurrence_df[species].sum()
            for species in species_cols
        },
        'detection_rate_per_species': {
            species: f"{100 * occurrence_df[species].sum() / len(occurrence_df):.1f}%"
            for species in species_cols
        }
    }

    return summary
