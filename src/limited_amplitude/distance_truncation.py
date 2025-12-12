"""
Distance truncation for birdsong detections using amplitude-based methods.

This module applies distance truncation on birdsong detections using the method
described in Lebeuf-Taylor et al., 2025. It maps species to reference species
and assigns distances based on amplitude measurements.

The workflow produces two outputs:
1. Amplitude dataframe: Species detections with mean amplitude values
2. Truncated dataframe: Amplitude-filtered detections within distance threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple


def load_predicted_amplitudes(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the predicted amplitudes for different distances.

    Args:
        file_path: Path to the predicted amplitudes CSV. If None, uses default
                  file from package data folder.

    Returns:
        DataFrame with predicted amplitudes
    """
    if file_path is None:
        # Use default predicted amplitudes from package data folder
        package_dir = Path(__file__).parent.parent.parent
        file_path = package_dir / "data" / "all_spp_predicted_amplitudes.csv"

    file_path = Path(file_path)
    print(f"Loading predicted amplitudes from {file_path}...")
    predicted_amps = pd.read_csv(file_path)
    print(f"Loaded {len(predicted_amps)} predicted amplitude records")
    return predicted_amps


def load_species_references(file_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """
    Load the mapping between species and reference species.

    Args:
        file_path: Path to the species and references CSV. If None, uses default
                  file from package data folder.

    Returns:
        Dictionary mapping species codes to reference species codes
    """
    if file_path is None:
        # Use default species references from package data folder
        package_dir = Path(__file__).parent.parent.parent
        file_path = package_dir / "data" / "species and references.csv"

    file_path = Path(file_path)
    print(f"Loading species-reference mapping from {file_path}...")
    spp_df = pd.read_csv(file_path)

    # Validate required columns
    if 'species' not in spp_df.columns or 'reference' not in spp_df.columns:
        raise ValueError(
            "Species reference file must contain 'species' and 'reference' columns")

    spp_dict = dict(zip(spp_df['species'], spp_df['reference']))
    print(f"Loaded {len(spp_dict)} species-reference mappings")
    return spp_dict


def validate_metadata(metadata_df: pd.DataFrame) -> bool:
    """
    Validate that a metadata DataFrame has the required columns and types.

    This function checks the metadata without modifying it. Use prepare_metadata()
    to fix type issues.

    Args:
        metadata_df: DataFrame to validate. Must contain 'location', 'canopy',
                    and 'SM2' columns.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If required columns are missing or types are invalid.
    """
    required_columns = ['location', 'canopy', 'SM2']
    missing_columns = [
        col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {missing_columns}")

    # Check if canopy and SM2 can be converted to numeric
    for col in ['canopy', 'SM2']:
        try:
            pd.to_numeric(metadata_df[col], errors='raise')
        except (ValueError, TypeError):
            raise ValueError(
                f"Column '{col}' contains non-numeric values that cannot be converted.")

    return True


def prepare_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare metadata DataFrame by ensuring correct column types.

    This function creates a copy of the input and fixes type issues:
    - Converts 'canopy' to numeric (NaN becomes 0)
    - Converts 'SM2' to numeric (NaN becomes 1)

    Args:
        metadata_df: DataFrame with 'location', 'canopy', and 'SM2' columns.

    Returns:
        A copy of the DataFrame with corrected types.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ['location', 'canopy', 'SM2']
    missing_columns = [
        col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {missing_columns}")

    metadata_df = metadata_df.copy()

    # Ensure numeric types
    metadata_df['canopy'] = pd.to_numeric(
        metadata_df['canopy'], errors='coerce').fillna(0)
    metadata_df['SM2'] = pd.to_numeric(
        metadata_df['SM2'], errors='coerce').fillna(1)

    return metadata_df


def calculate_mean_amplitude(
    left_amp: float,
    right_amp: float,
    handle_missing: str = 'use_available'
) -> float:
    """
    Calculate mean amplitude from left and right microphone values.

    Args:
        left_amp: Amplitude from left microphone
        right_amp: Amplitude from right microphone
        handle_missing: How to handle missing values:
                       'use_available' - use whichever mic is available
                       'require_both' - return NaN if either is missing
                       'mean_only' - only calculate mean if both present

    Returns:
        Mean amplitude value or NaN
    """
    left_valid = pd.notna(left_amp)
    right_valid = pd.notna(right_amp)

    if handle_missing == 'use_available':
        if left_valid and right_valid:
            return (left_amp + right_amp) / 2
        elif left_valid:
            return left_amp
        elif right_valid:
            return right_amp
        else:
            return np.nan
    elif handle_missing == 'require_both':
        if left_valid and right_valid:
            return (left_amp + right_amp) / 2
        else:
            return np.nan
    elif handle_missing == 'mean_only':
        if left_valid and right_valid:
            return (left_amp + right_amp) / 2
        else:
            return np.nan
    else:
        raise ValueError(f"Invalid handle_missing value: {handle_missing}")


def apply_mic_overrides(
    tags_df: pd.DataFrame,
    report_df: pd.DataFrame,
    left_amp_col: str = 'left_freq_filter_tag_peak_level_dbfs',
    right_amp_col: str = 'right_freq_filter_tag_peak_level_dbfs',
    task_comments_col: str = 'task_comments'
) -> pd.DataFrame:
    """
    Apply microphone override based on task comments in the report.

    When a location has a task comment indicating to use only one microphone,
    this function copies that mic's amplitude to both channels so the mean
    calculation effectively uses only that microphone.

    Recognized comments (case-insensitive):
    - "use right mic only": copies right mic value to left mic column
    - "use left mic only": copies left mic value to right mic column

    Args:
        tags_df: DataFrame with tag records containing amplitude columns.
        report_df: DataFrame with 'location' and task_comments_col columns.
        left_amp_col: Column name for left microphone amplitude.
        right_amp_col: Column name for right microphone amplitude.
        task_comments_col: Column name for task comments in report_df.

    Returns:
        A copy of tags_df with amplitude columns adjusted based on task comments.
    """
    tags_df = tags_df.copy()

    if task_comments_col not in report_df.columns:
        print(f"Warning: '{task_comments_col}' column not found in report. Skipping mic overrides.")
        return tags_df

    if 'location' not in report_df.columns:
        print("Warning: 'location' column not found in report. Skipping mic overrides.")
        return tags_df

    # Normalize task comments to lowercase
    report_df = report_df.copy()
    report_df[task_comments_col] = report_df[task_comments_col].fillna('').str.lower().str.strip()

    # Find locations where we should use only one mic
    use_right_only = report_df.loc[
        report_df[task_comments_col] == "use right mic only", 'location'
    ]
    use_left_only = report_df.loc[
        report_df[task_comments_col] == "use left mic only", 'location'
    ]

    # Apply overrides
    right_only_mask = tags_df['location'].isin(use_right_only)
    left_only_mask = tags_df['location'].isin(use_left_only)

    # For "use right mic only": copy right to left
    if right_only_mask.any():
        tags_df.loc[right_only_mask, left_amp_col] = tags_df.loc[right_only_mask, right_amp_col]
        print(f"Applied 'use right mic only' override to {right_only_mask.sum()} records")

    # For "use left mic only": copy left to right
    if left_only_mask.any():
        tags_df.loc[left_only_mask, right_amp_col] = tags_df.loc[left_only_mask, left_amp_col]
        print(f"Applied 'use left mic only' override to {left_only_mask.sum()} records")

    return tags_df


def create_amplitude_dataframe(
    tags_df: pd.DataFrame,
    recordings_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    report_df: Optional[pd.DataFrame] = None,
    left_amp_col: str = 'left_freq_filter_tag_peak_level_dbfs',
    right_amp_col: str = 'right_freq_filter_tag_peak_level_dbfs',
    filter_complete_only: bool = True,
    filter_vocalization: Optional[str] = 'Song',
    filter_task_status: Optional[str] = 'Transcribed'
) -> pd.DataFrame:
    """
    Create amplitude dataframe (Output 1) with mean amplitude values per species.

    This is the first output described in the README: a wide-format dataframe where
    columns are location, recording_date_time, and species codes, with values as
    mean amplitude.

    Args:
        tags_df: DataFrame with WildTrax tags data.
        recordings_df: DataFrame with WildTrax recordings data.
        metadata_df: DataFrame with location, canopy, and SM2 columns.
        report_df: Optional DataFrame with WildTrax report data. If provided and
                  contains a 'task_comments' column, microphone overrides will be
                  applied (e.g., "use left mic only", "use right mic only").
        left_amp_col: Column name for left microphone amplitude.
        right_amp_col: Column name for right microphone amplitude.
        filter_complete_only: Only include complete recordings.
        filter_vocalization: Filter to specific vocalization type (e.g., 'Song').
        filter_task_status: Filter to specific task status (e.g., 'Transcribed').

    Returns:
        DataFrame with location, recording_date_time, and species amplitude columns.
    """
    # Apply filters
    if filter_complete_only and 'is_complete' in tags_df.columns:
        tags_df = tags_df[tags_df['is_complete'].isin(
            [True, 't', 'T', 'true', 'True'])]
        print(f"Filtered to {len(tags_df)} complete records")

    if filter_vocalization and 'vocalization' in tags_df.columns:
        tags_df = tags_df[tags_df['vocalization'] == filter_vocalization]
        print(f"Filtered to {len(tags_df)} {filter_vocalization} records")

    if filter_task_status and 'aru_task_status' in tags_df.columns:
        tags_df = tags_df[tags_df['aru_task_status'] == filter_task_status]
        print(f"Filtered to {len(tags_df)} {filter_task_status} records")

    # Apply microphone overrides based on task comments if report is provided
    if report_df is not None:
        tags_df = apply_mic_overrides(
            tags_df, report_df, left_amp_col, right_amp_col
        )

    # Calculate mean amplitude for each tag
    if left_amp_col in tags_df.columns and right_amp_col in tags_df.columns:

        tags_df = tags_df[['location', 'recording_date_time', 'species_code', 'detection_time', 'task_duration',
                           'left_freq_filter_tag_peak_level_dbfs', 'right_freq_filter_tag_peak_level_dbfs']]
        tags_df['mean_amp'] = tags_df.apply(
            lambda row: calculate_mean_amplitude(
                row[left_amp_col],
                row[right_amp_col],
                handle_missing='use_available'
            ),
            axis=1
        )
    else:
        raise ValueError(
            f"Required amplitude columns not found: {left_amp_col}, {right_amp_col}")

    # Filter out rows with null amplitude
    tags_df = tags_df[tags_df['mean_amp'].notna()]

    tags_df['recording_date_time'] = pd.to_datetime(
        tags_df['recording_date_time'])

    # Fill gaps with recordings that have no detections
    recordings_df['recording_date_time'] = pd.to_datetime(
        recordings_df['recording_date_time'])

    recordings_df = recordings_df[['location', 'recording_date_time']]

    # Find recordings not in tags using anti-join
    not_in_tags_df = recordings_df.merge(
        tags_df[['location', 'recording_date_time']],
        on=['location', 'recording_date_time'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns='_merge')

    full_df = pd.concat([tags_df, not_in_tags_df], ignore_index=True)

    # Add metadata (canopy and SM2 status)
    full_df = pd.merge(
        full_df,
        metadata_df[['location', 'canopy', 'SM2']],
        on='location',
        how='left'
    )

    return full_df


def estimate_distance_from_amplitude(
    detections_df: pd.DataFrame,
    species_col: str = 'species_code',
    amplitude_col: str = 'mean_amp',
    canopy_col: str = 'canopy',
    sm2_col: str = 'SM2'
) -> pd.DataFrame:
    """
    Estimate distance for each detection based on its amplitude value.

    This function takes a DataFrame of detections (where each row is a unique detection
    for a given species) and uses the predicted_amps lookup table to estimate the
    distance based on the mean_amp value for each detection.

    Args:
        detections_df: DataFrame with detection records. Must contain columns for
                      species, amplitude, canopy, and SM2 status. Can be the output
                      of create_amplitude_dataframe.
        species_col: Column name for species code in detections_df (default: 'species_code').
        amplitude_col: Column name for mean amplitude in detections_df (default: 'mean_amp').
        canopy_col: Column name for canopy status in detections_df (default: 'canopy').
        sm2_col: Column name for SM2 status in detections_df (default: 'SM2').

    Returns:
        DataFrame with all original columns plus 'distance_est' column containing
        the estimated distance for each detection. Returns np.nan for detections
        where distance cannot be estimated (e.g., species not in reference mapping,
        amplitude outside predicted range).
    """
    print(f"Estimating distances for {len(detections_df)} detections...")
    predicted_amps = load_predicted_amplitudes()
    species_references = load_species_references()

    # Create a copy to avoid modifying the original
    result_df = detections_df.copy()
    result_df['_idx'] = range(len(result_df))  # Preserve original order

    # Map species to reference species
    result_df['_reference_spp'] = result_df[species_col].map(species_references)

    # Merge with predicted amplitudes to find nearest match
    merge_df = result_df[
        result_df['_reference_spp'].notna() & result_df[amplitude_col].notna()
    ].merge(
        predicted_amps[['target_spp', 'canopy', 'SM2', 'distance', 'predicted']],
        left_on=['_reference_spp', canopy_col, sm2_col],
        right_on=['target_spp', 'canopy', 'SM2'],
        how='left'
    )

    # Calculate amplitude difference and find nearest match per detection
    merge_df = merge_df.dropna()
    merge_df['_amp_diff'] = abs(merge_df['predicted'] - merge_df[amplitude_col])

    # Get the row with minimum amplitude difference for each original detection
    nearest = merge_df.loc[merge_df.groupby('_idx')['_amp_diff'].idxmin()]
    distance_map = nearest.set_index('_idx')['distance']

    # Map back to result
    result_df['distance_est'] = result_df['_idx'].map(distance_map)

    # Clean up temporary columns
    result_df = result_df.drop(columns=['_idx', '_reference_spp'])

    estimated_count = result_df['distance_est'].notna().sum()
    print(f"Estimated distances for {estimated_count} of {len(result_df)} detections")

    return result_df


def prepare_amplitude_thresholds(
    predicted_amps: pd.DataFrame,
    species_references: Dict[str, str],
    distance_threshold: float = 150.0
) -> Dict:
    """
    Prepare amplitude thresholds for each species at specified distance.

    Args:
        predicted_amps: DataFrame with predicted amplitudes
        species_references: Dictionary mapping species to reference species
        distance_threshold: Distance in meters for truncation

    Returns:
        Nested dictionary: {species: {canopy: {SM2_status: threshold}}}
    """
    print(f"Preparing amplitude thresholds for {distance_threshold}m...")

    # Work on a copy to avoid modifying the input DataFrame
    predicted_amps = predicted_amps.copy()

    # Find nearest distance to threshold for each combination
    predicted_amps['distance_to_threshold'] = abs(
        predicted_amps['distance'] - distance_threshold
    )
    nearest_indices = predicted_amps.groupby(['target_spp', 'canopy', 'SM2'])[
        'distance_to_threshold'
    ].idxmin()
    thresholds_at_distance = predicted_amps.loc[nearest_indices]

    # Create threshold dictionary
    thresholds_dict = {}

    for species, reference in species_references.items():
        ref_thresholds = thresholds_at_distance[
            thresholds_at_distance['target_spp'] == reference
        ]

        if not ref_thresholds.empty:
            thresholds_dict[species] = {}

            for _, row in ref_thresholds.iterrows():
                canopy_type = row['canopy']
                sm2_status = row['SM2']
                threshold = row['predicted']

                if canopy_type not in thresholds_dict[species]:
                    thresholds_dict[species][canopy_type] = {}

                thresholds_dict[species][canopy_type][f'SM2_{sm2_status}'] = threshold
        else:
            print(
                f"Warning: No thresholds found for reference species {reference} (for {species})")

    print(f"Created thresholds for {len(thresholds_dict)} species")
    return thresholds_dict


def apply_distance_truncation(
    tags_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    report_df: Optional[pd.DataFrame] = None,
    predicted_amps: Optional[Union[pd.DataFrame, str, Path]] = None,
    species_references: Optional[Union[Dict[str, str], str, Path]] = None,
    distance_threshold: float = 150.0,
    filter_complete_only: bool = True,
    filter_vocalization: Optional[str] = 'Song',
    left_amp_col: str = 'left_freq_filter_tag_peak_level_dbfs',
    right_amp_col: str = 'right_freq_filter_tag_peak_level_dbfs'
) -> pd.DataFrame:
    """
    Apply distance truncation to individual tag records.

    This function processes individual tags (not aggregated), applies amplitude thresholds
    based on distance, and returns only the tags that pass the threshold. Each tag retains
    its mean amplitude value.

    Args:
        tags_df: DataFrame with WildTrax tags data.
        metadata_df: DataFrame with location, canopy, and SM2 columns.
        report_df: Optional DataFrame with WildTrax report data. If provided and
                  contains a 'task_comments' column, microphone overrides will be
                  applied (e.g., "use left mic only", "use right mic only").
        predicted_amps: DataFrame with predicted amplitudes, or path to CSV file.
                       If None, uses default data/all_spp_predicted_amplitudes.csv
        species_references: Dictionary mapping species to reference species, or path to CSV file.
                           If None, uses default data/species and references.csv
        distance_threshold: Distance in meters for truncation (default: 150.0).
        filter_complete_only: Only include complete recordings.
        filter_vocalization: Filter to specific vocalization type (e.g., 'Song').
        left_amp_col: Column name for left microphone amplitude.
        right_amp_col: Column name for right microphone amplitude.

    Returns:
        DataFrame of individual tags that pass the distance threshold, with mean_amp column.
    """
    # Apply filters
    if filter_complete_only and 'is_complete' in tags_df.columns:
        tags_df = tags_df[tags_df['is_complete'].isin([True, 't', 'T', 'true', 'True'])]
        print(f"Filtered to {len(tags_df)} complete records")

    if filter_vocalization and 'vocalization' in tags_df.columns:
        tags_df = tags_df[tags_df['vocalization'] == filter_vocalization]
        print(f"Filtered to {len(tags_df)} {filter_vocalization} records")

    # Apply microphone overrides based on task comments if report is provided
    if report_df is not None:
        tags_df = apply_mic_overrides(
            tags_df, report_df, left_amp_col, right_amp_col
        )

    # Load predicted amplitudes if needed
    if predicted_amps is None or isinstance(predicted_amps, (str, Path)):
        predicted_amps = load_predicted_amplitudes(predicted_amps)

    # Load species references if needed
    if species_references is None or isinstance(species_references, (str, Path)):
        species_references = load_species_references(species_references)

    # Get thresholds for specified distance
    thresholds_dict = prepare_amplitude_thresholds(
        predicted_amps,
        species_references,
        distance_threshold
    )

    # Calculate mean amplitude for each tag
    if left_amp_col in tags_df.columns and right_amp_col in tags_df.columns:
        tags_df['mean_amp'] = tags_df.apply(
            lambda row: calculate_mean_amplitude(
                row[left_amp_col],
                row[right_amp_col],
                handle_missing='use_available'
            ),
            axis=1
        )
    else:
        raise ValueError(
            f"Required amplitude columns not found: {left_amp_col}, {right_amp_col}")

    # Filter out rows with null amplitude
    tags_df = tags_df[tags_df['mean_amp'].notna()]
    print(f"Retained {len(tags_df)} records with valid amplitude")

    # Merge with metadata
    metadata_subset = metadata_df[['location', 'canopy', 'SM2']].drop_duplicates(subset=['location'])
    tags_df = pd.merge(
        tags_df,
        metadata_subset,
        on='location',
        how='left'
    )

    # Apply distance truncation to individual tags
    tags_df['passes_threshold'] = False

    for species, canopy_thresholds in thresholds_dict.items():
        # Get tags for this species
        species_mask = tags_df['species_code'] == species

        for idx in tags_df[species_mask].index:
            row = tags_df.loc[idx]
            canopy_type = row['canopy']
            sm2_value = row['SM2']

            # Get threshold for this combination
            if canopy_type in canopy_thresholds:
                sm2_key = f'SM2_{int(sm2_value)}'
                if sm2_key in canopy_thresholds[canopy_type]:
                    threshold = canopy_thresholds[canopy_type][sm2_key]

                    # Check if amplitude exceeds threshold
                    if row['mean_amp'] > threshold:
                        tags_df.at[idx, 'passes_threshold'] = True

    # Filter to only tags that pass the threshold
    truncated_tags = tags_df[tags_df['passes_threshold']].copy()
    print(f"Retained {len(truncated_tags)} individual tags that pass {distance_threshold}m threshold")

    return truncated_tags


def convert_to_occurrence(
    truncated_tags: pd.DataFrame,
    recordings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert truncated tags to occurrence/occupancy format (1/0).

    This function:
    1. Takes individual tags that passed distance truncation
    2. Converts to binary presence/absence per species per recording
    3. Ensures ALL location + recording_date_time combinations from recordings DataFrame are included

    Args:
        truncated_tags: DataFrame of individual tags from apply_distance_truncation()
        recordings_df: DataFrame with all recordings (ground truth of all recordings).
                      Must contain 'location' and 'recording_date_time' columns.

    Returns:
        DataFrame with location, recording_date_time, and binary species occurrence (1/0)
    """
    print(f"Converting truncated tags to occurrence format...")

    # Get all unique location + recording_date_time combinations (ground truth)
    all_recordings = recordings_df[['location', 'recording_date_time']].drop_duplicates()
    print(f"Found {len(all_recordings)} unique location/recording combinations")

    # Get unique species detected (after truncation)
    species_list = truncated_tags['species_code'].unique().tolist()
    print(f"Processing {len(species_list)} species")

    # Create occurrence dataframe: 1 if species detected at location/recording, 0 otherwise
    occurrence_records = []

    for _, recording in all_recordings.iterrows():
        location = recording['location']
        recording_time = recording['recording_date_time']

        # Get species detected at this location/recording
        detected_species = truncated_tags[
            (truncated_tags['location'] == location) &
            (truncated_tags['recording_date_time'] == recording_time)
        ]['species_code'].unique()

        # Build row: location, recording_time, then 1/0 for each species
        row = {'location': location, 'recording_date_time': recording_time}
        for species in species_list:
            row[species] = 1 if species in detected_species else 0

        occurrence_records.append(row)

    occurrence_df = pd.DataFrame(occurrence_records)

    print(f"Final occurrence dataframe: {len(occurrence_df)} recordings × {len(species_list)} species")

    # Report summary
    total_detections = occurrence_df[species_list].sum().sum()
    print(f"Total detections: {total_detections}")

    return occurrence_df


def convert_to_counts(
    truncated_tags: pd.DataFrame,
    recordings_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert truncated tags to count format.

    This function:
    1. Takes individual tags that passed distance truncation
    2. Counts the number of detections per species per recording
    3. Ensures ALL location + recording_date_time combinations from recordings DataFrame are included

    Args:
        truncated_tags: DataFrame of individual tags from apply_distance_truncation()
        recordings_df: DataFrame with all recordings (ground truth of all recordings).
                      Must contain 'location' and 'recording_date_time' columns.

    Returns:
        DataFrame with location, recording_date_time, and species detection counts
    """
    print(f"Converting truncated tags to count format...")

    # Get all unique location + recording_date_time combinations (ground truth)
    all_recordings = recordings_df[['location', 'recording_date_time']].drop_duplicates()
    print(f"Found {len(all_recordings)} unique location/recording combinations")

    # Count detections per location, recording_date_time, and species
    count_df = truncated_tags.groupby(
        ['location', 'recording_date_time', 'species_code']
    ).size().reset_index(name='count')

    # Pivot to wide format
    count_wide = count_df.pivot_table(
        index=['location', 'recording_date_time'],
        columns='species_code',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Merge with all recordings to ensure complete coverage
    count_complete = all_recordings.merge(
        count_wide,
        on=['location', 'recording_date_time'],
        how='left'
    )

    # Fill NaN with 0 (recordings with no detections)
    species_cols = [col for col in count_complete.columns
                   if col not in ['location', 'recording_date_time']]
    count_complete[species_cols] = count_complete[species_cols].fillna(0).astype(int)

    print(f"Final count dataframe: {len(count_complete)} recordings × {len(species_cols)} species")

    # Report summary
    total_detections = count_complete[species_cols].sum().sum()
    print(f"Total detections: {total_detections}")

    return count_complete


def get_truncation_summary(
    amplitude_df: pd.DataFrame,
    truncated_df: pd.DataFrame
) -> Dict:
    """
    Get summary statistics comparing before/after truncation.

    Args:
        amplitude_df: Original amplitude dataframe
        truncated_df: Truncated amplitude dataframe

    Returns:
        Dictionary with summary statistics
    """
    species_cols = [col for col in amplitude_df.columns
                    if col not in ['location', 'recording_date_time']]

    original_detections = amplitude_df[species_cols].notna().sum().sum()
    truncated_detections = truncated_df[species_cols].notna().sum().sum()
    removed_detections = original_detections - truncated_detections

    summary = {
        'total_recordings': len(amplitude_df),
        'total_species': len(species_cols),
        'original_detections': int(original_detections),
        'truncated_detections': int(truncated_detections),
        'removed_detections': int(removed_detections),
        'removal_rate': f"{100 * removed_detections / original_detections:.1f}%" if original_detections > 0 else "0%",
        'detections_per_species_original': {
            species: int(amplitude_df[species].notna().sum())
            for species in species_cols
        },
        'detections_per_species_truncated': {
            species: int(truncated_df[species].notna().sum())
            for species in species_cols
        }
    }

    print("\nTruncation Summary:")
    print(f"  Total recordings: {summary['total_recordings']}")
    print(f"  Total species: {summary['total_species']}")
    print(f"  Original detections: {summary['original_detections']}")
    print(f"  Truncated detections: {summary['truncated_detections']}")
    print(
        f"  Removed detections: {summary['removed_detections']} ({summary['removal_rate']})")

    return summary
