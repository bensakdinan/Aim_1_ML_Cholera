def convert_str_to_numeric(merged_cholera_data) -> None:
    """Converts 'Nature_of_Stool' to numeric type
    Non-Watery = 0.0
    Watery = 1.0
    Rice-Watery = 2.0
    Loose-Watery = 3.0
    """

    mapping = {
        'Non-Watery': 0.0,
        'Watery': 1.0,
        'Rice Watery': 2.0,
        'Loose Watery': 3.0
    }

    # Apply mapping
    merged_cholera_data['Nature_of_Stool'] = merged_cholera_data['Nature_of_Stool'].map(mapping)

    # Handle unknown values
    if merged_cholera_data['Nature_of_Stool'].isna().any():
        print("Warning: Some values in 'Nature_of_Stool' were not recognized and set to NaN")

    return merged_cholera_data
