import numpy as np
from reading_files.feature_selection import generate_stats_name_from_feature_name, eICU_time_features, eICU_nontime_features, mimic_time_features, mimic_nontime_features, lower

eICU_time_features = list(map(lower, eICU_time_features))
mimic_time_features = list(map(lower, mimic_time_features))

def seperate_based_on_feature(in_features_df, feature_to_seperate, threshold, in_labels, mimic_version='iv', reverse=False):

    """
    Separate input data to ID/OOD based on a specified feature and threshold.

    Parameters:
    -----------
    in_features_df: pd.DataFrame
        Input data as a pandas DataFrame.
    feature_to_separate: str
        The feature to separate the data on ('age', 'gender', 'ethnicity', 'admission_type', 'first_careunit').
    threshold: int or str
        The threshold or value to use for separation.
    in_labels: pd.Series
        Labels associated with the input data.
    mimic_version: str, optional
        The version of the MIMIC dataset ('iii' or 'iv'). Default is 'iv'.
    reverse: bool, optional
        If True, reverse the separation.
        Default is False.

    Returns:
    --------
    Tuple of NumPy arrays:
        - The first element contains in-distribution data.
        - The second element contains out-of-distribution (ood) data.
        - The third element contains labels corresponding to the in-distribution data.

    Raises:
    -------
    ValueError:
        If `feature_to_separate` is not one of the supported features ('age', 'gender', 'ethnicity', 'admission_type', 'first_careunit').
    """

    if feature_to_seperate == 'age':
        in_features_np = in_features_df.loc[in_features_df[feature_to_seperate] >= threshold].drop(columns=feature_to_seperate).to_numpy(dtype='float32')
        ood_features_np = in_features_df.loc[in_features_df[feature_to_seperate] < threshold].drop(columns=feature_to_seperate).to_numpy(dtype='float32')
        in_labels_np = in_labels.loc[in_features_df[feature_to_seperate] >= threshold].to_numpy().reshape(-1)
        in_labels_np_reverse = in_labels.loc[in_features_df[feature_to_seperate] < threshold].to_numpy().reshape(-1)

    elif feature_to_seperate == 'gender':
        in_features_np = in_features_df.loc[in_features_df[feature_to_seperate] == threshold].drop(columns=feature_to_seperate).to_numpy(dtype='float32')
        ood_features_np = in_features_df.loc[in_features_df[feature_to_seperate] != threshold].drop(columns=feature_to_seperate).to_numpy(dtype='float32')
        in_labels_np = in_labels.loc[in_features_df[feature_to_seperate] == threshold].to_numpy().reshape(-1)
        in_labels_np_reverse = in_labels.loc[in_features_df[feature_to_seperate] != threshold].to_numpy().reshape(-1)

    elif feature_to_seperate == 'ethnicity': #only eicu
        ethnicity = ['Native American', 'African American', 'Asian', 'Caucasian', 'Hispanic']
        in_features_np = in_features_df.loc[in_features_df[threshold] == 1].drop(columns=ethnicity).to_numpy(dtype='float32')
        ood_features_np = in_features_df.loc[in_features_df[threshold] == 0].drop(columns=ethnicity).to_numpy(dtype='float32')
        in_labels_np = in_labels.loc[in_features_df[threshold] == 1].to_numpy().reshape(-1)
        in_labels_np_reverse = in_labels.loc[in_features_df[threshold] == 0].to_numpy().reshape(-1)

    elif feature_to_seperate == 'admission_type': #only mimic
        if mimic_version == 'iii':
          admission_type = ['EMERGENCY' 'ELECTIVE' 'URGENT']
        elif mimic_version == 'iv':
          admission_type = ['EU OBSERVATION', 'EW EMER.', 'OBSERVATION ADMIT', 'URGENT', 'DIRECT EMER.', 'AMBULATORY OBSERVATION', 'DIRECT OBSERVATION', 'SURGICAL SAME DAY ADMISSION', 'ELECTIVE']
        in_features_np = in_features_df.loc[in_features_df[threshold] == 1].drop(columns=admission_type).to_numpy(dtype='float32')
        ood_features_np = in_features_df.loc[in_features_df[threshold] == 0].drop(columns=admission_type).to_numpy(dtype='float32')
        in_labels_np = in_labels.loc[in_features_df[threshold] == 1].to_numpy().reshape(-1)
        in_labels_np_reverse = in_labels.loc[in_features_df[threshold] == 0].to_numpy().reshape(-1)

    elif feature_to_seperate == 'first_careunit': #only mimic
        if mimic_version == 'iii':
          first_careunit = ['MICU' 'SICU' 'CCU' 'CSRU' 'TSICU']
        elif mimic_version == 'iv':
          first_careunit = ['MICU', 'CCU', 'MICU/SICU', 'Neuro Intermediate', 'SICU', 'CVICU', 'Neuro SICU', 'TSICU', 'Neuro Stepdown']
        in_features_np = in_features_df.loc[in_features_df[threshold] == 1].drop(columns=first_careunit).to_numpy(dtype='float32')
        ood_features_np = in_features_df.loc[in_features_df[threshold] == 0].drop(columns=first_careunit).to_numpy(dtype='float32')
        in_labels_np = in_labels.loc[in_features_df[threshold] == 1].to_numpy().reshape(-1)
        in_labels_np_reverse = in_labels.loc[in_features_df[threshold] == 0].to_numpy().reshape(-1)

    if reverse:
        return ood_features_np, in_features_np, in_labels_np_reverse
    else:
        return in_features_np, ood_features_np, in_labels_np


def other_domain(in_time_features, in_nontime_features, ood_time_features, ood_nontime_features, in_features_df, ood_features_df):
    
    """
    This function gets ID and OOD data coming from different datasets and keeps the joint features between them in the same order.

    Parameters:
    -----------
    in_time_features: list of str
        List of time-based features for the in-distribution data.
    in_nontime_features: list of str
        List of non-time-based features for the in-distribution data.
    ood_time_features: list of str
        List of time-based features for the out-of-distribution data.
    ood_nontime_features: list of str
        List of non-time-based features for the out-of-distribution data.
    in_features_df: pd.DataFrame
        DataFrame containing in-distribution features.
    ood_features_df: pd.DataFrame
        DataFrame containing out-of-distribution features.

    Returns:
    --------
    Tuple of NumPy arrays:
        - The first element contains in-distribution features.
        - The second element contains out-of-distribution features.
    """
    
    mapping= {'Temperature':'Temperature (C)', 'Oxygen saturation': 'O2 Saturation', 'Mean blood pressure':'MAP (mmHg)', 'glascow coma scale total':'GCS Total',
               'Systolic blood pressure':'Invasive BP Systolic', 'Diastolic blood pressure':'Invasive BP Diastolic', 'FiO2':'Fraction of inspired oxygen'}
    mapping = dict(map(lambda x: (x[0].lower(), x[1].lower()), mapping.items()))
    inv_map = {v: k for k, v in mapping.items()}
    mapping.update(inv_map)

    hold_in = []
    hold_out = []

    for name_in in in_time_features:
        if name_in in mapping.keys():
           name_ood= mapping[name_in]
        else:
          name_ood = name_in

        if name_ood in ood_time_features:
            adding_features = []
            hold_out += generate_stats_name_from_feature_name(name_ood)
            hold_in += generate_stats_name_from_feature_name(name_in)

    for name_in in in_nontime_features:
        if name_in in ood_nontime_features:
            hold_in.append(name_in)
            hold_out.append(name_in)

    in_features_np = in_features_df[hold_in].to_numpy(dtype='float32')
    ood_features_np = ood_features_df[hold_out].to_numpy(dtype='float32')

    return in_features_np, ood_features_np


def feature_multplication(in_features_df): 
    """
    Generate feature multiplication settings for evaluation.

    This function selects a random set of features from the input DataFrame for multiplication with different scales during evaluation. 
    The multiplication algorithm itself is done during evaluation.

    Parameters:
    -----------
    in_features_df: pd.DataFrame
        Input data as a pandas DataFrame.

    Returns:
    --------
    Tuple of NumPy arrays and lists:
        - The first element contains in-distribution features as a NumPy array.
        - The second element is a list of scales to apply for feature multiplication.
        - The third element is a list of randomly selected feature indices that each time one of them would be used for scaling.
    """

    N_FEATURES = 100
    SCALES = [1, 10, 100, 1000]

    in_features_np = in_features_df.to_numpy(dtype='float32')
    random_sample = np.random.choice(
        np.arange(0, in_features_np.shape[1]), N_FEATURES, replace=False)

    return in_features_np, SCALES, random_sample


def get_params_data(in_distribution, in_features_df, in_label_df, ood_type, ood_features_df=None, feature_to_seperate=None, threshold=None, mimic_version='iv'):
    """
    Prepare the ID and OOD data based on specified settings.

    Parameters:
    -----------
    in_distribution: str
        The distribution of in-distribution data ('eicu' or 'mimic').
    in_features_df: pd.DataFrame
        DataFrame containing in-distribution features.
    in_label_df: pd.DataFrame
        DataFrame containing in-distribution labels.
    ood_type: str
        The type of out-of-distribution data ('other_domain', 'feature_separation', or 'multiplication').
    ood_features_df: pd.DataFrame, optional
        DataFrame containing out-of-distribution features. Required when ood_type is 'other_domain'.
    feature_to_separate: str, optional
        The feature used for separation. Required when ood_type is 'feature_separation'.
    threshold: float or str, optional
        The threshold or value used for separation. Required when ood_type is 'feature_separation'.
    mimic_version: str, optional
        The version of the MIMIC dataset ('iii' or 'iv'). Default is 'iv'.

    Returns:
    --------
    Tuple of NumPy arrays and lists:
        Depending on the specified ood_type, the returned tuple may include:
        - in-distribution features as a NumPy array.
        - out-of-distribution features as a NumPy array.
        - in-distribution labels as a NumPy array.
        - Additional elements may be included based on the ood_type.
    """  
    
    if in_distribution == 'eicu':
        in_time_features, in_nontime_features = eICU_time_features, eICU_nontime_features
    elif in_distribution == 'mimic':
        in_time_features, in_nontime_features = mimic_time_features, mimic_nontime_features

    if ood_type == 'other_domain':
        if in_distribution == 'eicu':
            ood_time_features, ood_nontime_features = mimic_time_features, mimic_nontime_features
        elif in_distribution == 'mimic':
            ood_time_features, ood_nontime_features = eICU_time_features, eICU_nontime_features
        in_features_np, ood_features_np = other_domain(in_time_features, in_nontime_features, ood_time_features, ood_nontime_features, in_features_df, ood_features_df)
        in_label_np = in_label_df.to_numpy().reshape(-1)
        return in_features_np, ood_features_np, in_label_np

    elif ood_type == 'feature_seperation':
        in_features_np, ood_features_np, in_label_np = seperate_based_on_feature(in_features_df, feature_to_seperate, threshold, in_label_df, mimic_version=mimic_version)
        return in_features_np, ood_features_np, in_label_np

    elif ood_type == 'multiplication':
        in_features_np, scales, random_sample = feature_multplication(in_features_df)
        in_label_np = in_label_df.to_numpy().reshape(-1)
        return in_features_np, scales, random_sample, in_label_np
