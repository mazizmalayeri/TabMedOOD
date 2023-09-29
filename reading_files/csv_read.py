import pandas as pd


def read(files_path='', mimic_version='iv'):
    """
    Load feature data from CSV files.

    Parameters:
    -----------
    files_path: str, optional
        The path to the directory where the CSV files are located. Default is an empty string.
    mimic_version: str, optional
        The version of the MIMIC dataset to load ('iii' or 'iv'). Default is 'iv'.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame
        Two pandas DataFrames containing feature data: eICU_features and mimic_features.
    """
    
    print('Loading eICU_features ...')
    eICU_features = pd.read_csv('eICU_preprocessed.csv').drop(columns=['Unnamed: 0'])
    eICU_features.columns = eICU_features.columns.str.lower()
    
    print('Loading mimic_features ...')
    if mimic_version == 'iii':
        mimic_path ='MIMIC_preprocessed.csv'
    elif mimic_version == 'iv':
        mimic_path = 'MIMIC_iv_preprocessed.csv'
    mimic_features = pd.read_csv(mimic_path).drop(columns=['Unnamed: 0']) 
    mimic_features.columns = mimic_features.columns.str.lower()
    
    return eICU_features, mimic_features
 

'''
* eICU
params_time_dependent = ['pH', 'Temperature (C)', 'Respiratory Rate', 'O2 Saturation', 'MAP (mmHg)', 'Heart Rate', 'glucose', 'GCS Total', 'Motor', 'Eyes', 'Verbal', 'FiO2', 'Invasive BP Diastolic', 'Invasive BP Systolic']
params_time_independent = ['patientunitstayid', 'gender', 'age', 'ethnicity', 'admissionheight', 'admissionweight', 'hospitaldischargestatus', 'unitdischargestatus']

* MIMIC
params_time_independent = ['subject_id', 'gender', 'ethnicity', 'age', 'diagnosis_at_admission', 'discharge_location', 'admission_type', 'first_careunit', 'mort_icu', 'mort_hosp', 'hospital_expire_flag', 'max_hours', 'readmission_30']
params_time_dependent = [alanine aminotransferase, albumin, albumin ascites, albumin pleural, albumin urine, alkaline phosphate, anion gap, asparate aminotransferase, basophils, bicarbonate, bilirubin, blood urea nitrogen, co2, co2 (etco2, pco2, etc.), calcium, calcium ionized, calcium urine, cardiac index, cardiac output thermodilution, cardiac output fick, central venous pressure, chloride, chloride urine, cholesterol, cholesterol hdl, cholesterol ldl, creatinine, creatinine ascites, creatinine body fluid, creatinine pleural, creatinine urine, diastolic blood pressure, eosinophils, fibrinogen, fraction inspired oxygen, fraction inspired oxygen set, glascow coma scale total, glucose, heart rate, height, hematocrit, hemoglobin, lactate, lactate dehydrogenase, lactate dehydrogenase pleural, lactic acid, lymphocytes, lymphocytes ascites, lymphocytes atypical, lymphocytes atypical csl, lymphocytes body fluid, lymphocytes percent, lymphocytes pleural, magnesium, mean blood pressure, mean corpuscular hemoglobin, mean corpuscular hemoglobin concentration, mean corpuscular volume, monocytes, monocytes csl, neutrophils, oxygen saturation, partial pressure of carbon dioxide, partial pressure of oxygen, partial thromboplastin time, peak inspiratory pressure, phosphate, phosphorous, plateau pressure, platelets, positive end-expiratory pressure, positive end-expiratory pressure set, post void residual, potassium, potassium serum, prothrombin time inr, prothrombin time pt, pulmonary artery pressure mean, pulmonary artery pressure systolic, pulmonary capillary wedge pressure, red blood cell count, red blood cell count csf, red blood cell count ascites, red blood cell count pleural, red blood cell count urine, respiratory rate, respiratory rate set, sodium, systemic vascular resistance, systolic blood pressure, temperature, tidal volume observed, tidal volume set, tidal volume spontaneous, total protein, total protein urine, troponin-i, troponin-t, venous pvo2, weight, white blood cell count, white blood cell count urine, ph, ph urine]

***Note:***

**eICU:** We have all suggested timely features of trust issues in our features (14 from 14).
['pH', 'Temperature (C)', 'Respiratory Rate', 'O2 Saturation', 'MAP (mmHg)', 'Heart Rate', 'glucose', 'GCS Total', 'Motor', 'Eyes', 'Verbal', 'FiO2', 'Invasive BP Diastolic', 'Invasive BP Systolic']

 **mimic:** Only 11 of 14 featurs suggested in trust issues paper is available.
['pH', 'Temperature', 'Respiratory rate', 'Oxygen saturation', 'Mean blood pressure', 'Heart Rate', 'Glucose', 'glascow coma scale total', 'Fraction inspired oxygen', 'Systolic blood pressure', 'Diastolic blood pressure']

***Note:*** There are a lot of missing values in these 11 featrures of MIMIC. From 16976 samples, these are the number of each features missing values:
0 6081 / 1 148 / 2 118 / 3 71 / 4 114 / 5 114 / 6 154 / 7 6804 / 8 12301 / 9 114 / 10 114
So you may like to ignore the 8th feature (FIO) in your analysis.
'''

def check(eICU_features, mimic_features, mimic_version='iv'):
    """
    Print and inspect the loaded feature data from eICU and MIMIC datasets.

    Parameters:
    -----------
    eICU_features: pd.DataFrame
        The feature data from the eICU dataset.
    mimic_features: pd.DataFrame
        The feature data from the MIMIC dataset.
    mimic_version: str, optional
        The version of the MIMIC dataset loaded ('iii' or 'iv'). Default is 'iv'.
    """
    
    print('eICU_features head')
    print(eICU_features.head())
    print('###########################')
    
    print('mimic_features head')
    print(mimic_features.head())
    print('###########################')  
    
    print('eICU unique genders:', eICU_features['gender'].unique())
    print('eICU unique ethnicity:', eICU_features['ethnicity'].unique())
    print('###########################') 
    
    print('MIMIC unique genders:', mimic_features['gender'].unique())
    print('MIMIC unique ethnicity:', mimic_features['ethnicity'].unique())
    print('###########################') 
    
    print('Some features in MIMIC:')
    print(mimic_features['admission_type'].unique())
    print(mimic_features['first_careunit'].unique())
    if mimic_version == 'iii':
        print(mimic_features['readmission_30'].unique())
        print(mimic_features['diagnosis_at_admission'].unique())
    print('###########################') 
