import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import time
import dask.dataframe as dd

def load_filtered_csv_dask(csv_path, patients_selected):
    start_time = time.time()

    # Load the large CSV file using dask
    usecols = ['SUBJECT_ID', 'HADM_ID', 'VALUENUM', 'VALUEUOM', 'ITEMID']

    # Specify dtypes for the columns
    dtypes = {
        'SUBJECT_ID': 'int64',
        'HADM_ID': 'int64',
        'VALUENUM': 'float64',
        'VALUEUOM': 'object',
        'ITEMID': 'int64'
    }

    # Load the large CSV file using dask with specified dtypes
    df = dd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
    # Convert patients_selected to a dask DataFrame
    patients_selected_dask = dd.from_pandas(patients_selected, npartitions=1)
    # Merge the dask DataFrames on 'SUBJECT_ID' and 'HADM_ID'
    filtered_df = df.merge(patients_selected_dask, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    # Compute the result and convert to a pandas DataFrame
    result = filtered_df.compute()
    end_time = time.time()
    print(f"Time taken to load and filter CSV: {end_time - start_time} seconds")
    return result

def clean_adm(df_adm, df_patient):
    df_adm = pd.merge(df_patient[['SUBJECT_ID', 'GENDER', 'DOB']], df_adm, on='SUBJECT_ID')
    df_adm['ADMITTIME'] = pd.to_datetime(df_adm['ADMITTIME'])
    df_adm['DOB'] = pd.to_datetime(df_adm['DOB'])
    df_adm['ADMITTIME_YEAR'] = df_adm['ADMITTIME'].dt.year
    df_adm['DOB_YEAR'] = df_adm['DOB'].dt.year

    # Directly subtract the years to get an initial age
    df_adm['AGE'] = df_adm['ADMITTIME_YEAR'] - df_adm['DOB_YEAR']
    # Adjust for month and day
    df_adm['AGE'] = df_adm.apply(lambda x: x['AGE'] - 1 if (x['ADMITTIME'].month, x['ADMITTIME'].day) < (x['DOB'].month, x['DOB'].day) else x['AGE'], axis=1)
    df_adm.drop(['ADMITTIME_YEAR', 'DOB_YEAR'], axis=1, inplace=True)
    # Cap all ages greater than 100 to 100
    df_adm['AGE'] = df_adm['AGE'].clip(upper=100)
    ## Remove if ADMISSION_TYPE is 'NEWBORN' or 'ELECTIVE'
    df_adm = df_adm[df_adm['ADMISSION_TYPE'] != 'NEWBORN']
    df_adm = df_adm[df_adm['ADMISSION_TYPE'] != 'ELECTIVE']
    '''
    Convert Strings to Dates.
    When converting dates, it is safer to use a datetime format.
    Setting the errors = 'coerce' flag allows for missing dates
    but it sets it to NaT (not a datetime)  when the string doesn't match the format.
    '''
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    ## Calculate duration in days as the difference between DISCHTIME and ADMITTIME
    df_adm['DURATION'] = (df_adm.DISCHTIME - df_adm.ADMITTIME).dt.days
    ## Keep only SUBJECT_ID, GENDER, AGE, HADM_ID, DURATION, ADMISSION_TYPE, ADMISSION_LOCATION, DISCHARGE_LOCATION, INSURANCE, RELITION, MARITAL_STATUS, ETHNICITY, DIAGNOSIS
    df_adm = df_adm[['SUBJECT_ID', 'GENDER', 'AGE', 'HADM_ID', 'DURATION', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'DIAGNOSIS']]
    return df_adm 

def refine_and_extract_icd9_category(code, icd9_categories):
    code = str(code)
    if code.startswith('E') or code.startswith('V'):
        return "external causes of injury and supplemental classification"
    try:
        # Handle regular numeric codes
        code_num = int(code[:3])  # Extract the first three digits and convert to integer
    except ValueError:
        return "Unknown category"

    for category in icd9_categories:
        if category["range_start"] <= code_num <= category["range_end"]:
            return category["category"]

    return "Unknown category"

def sample_patients(df, categories, total_cases):
    sampled_data = pd.DataFrame()
    # Calculate the proportion of each category in the original data
    category_proportions = df[df['SEQ_NUM'] == 1]['CATEGORY'].value_counts(normalize=True)
    for category in categories:
        category_df = df[(df['CATEGORY'] == category) & (df['SEQ_NUM'] == 1)]
        category_proportion = category_proportions.get(category, 0)
        # Calculate the number of samples for this category
        n_samples = int(total_cases * category_proportion)
        if len(category_df) >= n_samples:
            sampled_category_df = category_df.sample(n=n_samples, random_state=1)
        else:
            sampled_category_df = category_df

        sampled_data = pd.concat([sampled_data, sampled_category_df])

    # If the total sampled cases are less than the required total, randomly sample the remaining cases
    if len(sampled_data) < total_cases:
        remaining_cases = total_cases - len(sampled_data)
        remaining_sample = df[(df['SEQ_NUM'] == 1) & (~df.index.isin(sampled_data.index))].sample(n=remaining_cases, random_state=1)
        sampled_data = pd.concat([sampled_data, remaining_sample])

    return sampled_data


def patient_selection(diagnosis, ICD, admissions, icd9_categories, total_cases):
    ICD = ICD.drop(columns=["ROW_ID"])
    ## Merge on ICD9_CODE
    diagnosis = pd.merge(diagnosis, ICD, on="ICD9_CODE", how="left")
    ## Select if individual have less than 5 diagnosis
    diagnosis = diagnosis.groupby("SUBJECT_ID").filter(lambda x: len(x) < 5)
    ## Subset if contains keyword "diabetes" not case sensitive
    # Fill NaN values with an empty string
    diagnosis["LONG_TITLE"] = diagnosis["LONG_TITLE"].fillna("")
    ## Merge diagnosis on admission on SUBJECT_ID and HADM_ID
    diagnosis = pd.merge(diagnosis, admissions[["SUBJECT_ID", "HADM_ID", "AGE"]], on=["SUBJECT_ID", "HADM_ID"])
    diagnosis['CATEGORY'] = diagnosis['ICD9_CODE'].apply(
            lambda code: refine_and_extract_icd9_category(code, icd9_categories)
        )
    ## Choose only diagnosis with AGE > 18
    diagnosis = diagnosis[diagnosis["AGE"] > 18]
    values = diagnosis['CATEGORY'].value_counts()
    ## Select top 10 most frequent categories in values
    most_frequent_classes = values.nlargest(len(diagnosis['CATEGORY'].unique())).index.tolist()
    # Apply the refined function to the DataFrame
    sampled_patients = sample_patients(diagnosis, most_frequent_classes, total_cases)
    ## Select the unique set of SUBJECT_ID and HADM_ID
    sampled_patients = sampled_patients.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID'])
    ## Rename CATEGORY to DIAGNOSIS_CATEGORY
    patients_selected = sampled_patients.rename(columns={'CATEGORY': 'DIAGNOSIS_CATEGORY'})
    return patients_selected

def clean_vitals(df_chart, items):
    df_chart = pd.merge(df_chart, items, on='ITEMID')
    ## Select only Routine Vital Signs in CATEGORY
    df_chart = df_chart[df_chart['CATEGORY'] == 'Routine Vital Signs']
    ## Select SUBJECT_ID, HADM_ID, CHARTTIME, VALUE, VALUEUOM, LABEL
    df_chart = df_chart[['SUBJECT_ID', 'HADM_ID', 'VALUENUM', 'VALUEUOM', 'LABEL']]
    ## Group by SUBJECT_ID, HADM_ID, VALUEUOM and LABEL, and calculate median of the VALUENUM
    df_chart = df_chart.groupby(['SUBJECT_ID', 'HADM_ID', 'LABEL', 'VALUEUOM']).median()
    ## Reset the row index
    df_chart = df_chart.reset_index()
    ## Replace "?F" and "?C" in VALUEUOM with "F" and "C"
    df_chart['VALUEUOM'] = df_chart['VALUEUOM'].replace({'?F': 'F', '?C': 'C'})
    return df_chart

def clean_notes(df_notes):
    df_notes['CHARTDATE'] = pd.to_datetime(df_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')
    df_discharge = df_notes[df_notes['CATEGORY'] == 'Discharge summary']
    return df_discharge

## Clean to extract only paragraph on Chief Complaint, history of present illness
# Define functions to extract specific sections with variations in titles
def extract_section(text, section_keywords):
    pattern = r'(' + '|'.join(section_keywords) + r'):\s*(.*?)(?=\n[A-Z ]+[:]|$)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        return ' '.join([match[1].strip() for match in matches])
    return None

def extract_sections_from_summary(summary):
    chief_complaint = extract_section(summary, ["chief complaint", "reason for admission"])
    history_of_present_illness = extract_section(summary, ["history of present illness"])
    past_medical_history = extract_section(summary, ["past medical history"])
    review_of_system = extract_section(summary, ["review of system"])
    allergies = extract_section(summary, ["allergies"])
    social_history = extract_section(summary, ["social history"])
    family_history = extract_section(summary, ["family history"])
    return pd.Series({
        "CHIEF COMPLAINT": chief_complaint,
        "HISTORY OF PRESENT ILLNESS": history_of_present_illness,
        "PAST MEDICAL HISTORY": past_medical_history,
        "REVIEW OF SYSTEM": review_of_system,
        "ALLERGIES": allergies,
        "SOCIAL HISTORY": social_history,
        "FAMILY HISTORY": family_history
    })

def create_symptom_dataframe(results_df):
    def parse_symptoms_data(cell_data):
        # If the word "Symptom" is not found within cell data, return ""
        if "Symptom" not in cell_data:
            return ""

        # Split the string into individual symptom entries
        symptom_entries = re.findall(r"\{.*?\}", cell_data)

        # List to hold the parsed data
        parsed_data = []

        # Extract values for each symptom entry
        for entry in symptom_entries:


            # symptom = re.search(r"'Symptom': '([^']*)'", entry).group(1).replace('<N/A>', '')
            # duration = re.search(r"'Duration': '([^']*)'", entry).group(1).replace('<N/A>', '')
            # frequency = re.search(r"'Frequency': '([^']*)'", entry).group(1).replace('<N/A>', '')
            # intensity = re.search(r"'Intensity': '([^']*)'", entry).group(1).replace('<N/A>', '')
            # negation = re.search(r"'Negation': '([^']*)'", entry).group(1).replace('<N/A>', '')

            ## In case not all fields are presented
            symptom_match = re.search(r"'Symptom': '([^']*)'", entry)
            symptom = symptom_match.group(1).replace('<N/A>', '') if symptom_match else ''
            duration_match = re.search(r"'Duration': '([^']*)'", entry)
            duration = duration_match.group(1).replace('<N/A>', '') if duration_match else ''
            frequency_match = re.search(r"'Frequency': '([^']*)'", entry)
            frequency = frequency_match.group(1).replace('<N/A>', '') if frequency_match else ''
            intensity_match = re.search(r"'Intensity': '([^']*)'", entry)
            intensity = intensity_match.group(1).replace('<N/A>', '') if intensity_match else ''
            negation_match = re.search(r"'Negation': '([^']*)'", entry)
            negation = negation_match.group(1).replace('<N/A>', '') if negation_match else ''

            parsed_data.append({
                'Symptom': symptom,
                'Duration': duration,
                'Frequency': frequency,
                'Intensity': intensity,
                'Negation': negation
            })

        return parsed_data

    # Apply the function to each cell and expand the rows
    expanded_rows = []

    for idx, row in results_df.iterrows():
        parsed_data = parse_symptoms_data(row['SYMPTOMS'])
        for item in parsed_data:
            expanded_rows.append({
                'SUBJECT_ID': row['SUBJECT_ID'],
                'HADM_ID': row['HADM_ID'],
                **item
            })

    # Create a new DataFrame with the expanded rows
    df_symptom = pd.DataFrame(expanded_rows)
    return df_symptom

def create_history_dataframe(results_df):
    def extract_history(row):
        if not isinstance(row, str):
            return ""
        row = row.replace('[N]', '').replace('[n]', '')
        # Extract the history part
        start = row.find("'Answer':") + len("'Answer': ")
        end = row.find("}<", start)
        start = row.find("'Answer':") + len("'Answer': ")
        end = row.find("}<", start)
        history_string = row[start:end].strip()

        if (history_string == "<N/A?") or history_string == "<N/A>":
            return ""

        history_list = history_string.split('; ')
        return history_list

    # Apply the function to the DataFrame and create a new column
    results_df['extracted_history'] = results_df['MEDICAL HISTORY'].apply(lambda x: extract_history(x) if x else x)
    # keep only relevant
    history_df = results_df[['SUBJECT_ID', 'HADM_ID', 'extracted_history']]
    df_history = results_df[['SUBJECT_ID', 'HADM_ID', 'extracted_history']].explode('extracted_history').rename(columns={'extracted_history': 'Medical History'})
    return df_history

def create_allergies_dataframe(results_df):
    ## Create Allergies df
    def extract_allergies(row):
        # Find the content between 'Answer': and a closing tag that includes the word "allergies"
        start = row.find("'Answer':") + len("'Answer': ")
        end_pattern = re.compile(r"}<[^>]*allergies[^>]*>", re.IGNORECASE)
        end_match = end_pattern.search(row, start)

        if end_match:
            end = end_match.start()
            allergies_string = row[start:end].strip()
            if allergies_string == "<N/A>":
                return ""
            allergies_list = allergies_string.split('; ')
            return allergies_list
        return ""

    # Apply the function to the DataFrame and create a new column
    results_df["extracted_allergies"] = results_df['ALLERGIES'].apply(lambda x: extract_allergies(x) if x else x)
    df_allergies = results_df[['SUBJECT_ID', 'HADM_ID', 'extracted_allergies']].explode('extracted_allergies').rename(columns={'extracted_allergies': 'Allergies'})
    # Drop if empty in the Allergies column
    df_allergies = df_allergies[df_allergies['Allergies'] != '']
    return df_allergies

def create_socialhistory_dataframe(results_df):
    ## Create Social history df
    def extract_socialhistory(row):
        if isinstance(row, str):
            start = row.find("'Answer':") + len("'Answer': ")
            end = row.find("}<", start)
            socialhistory_string = row[start:end].strip()
            if socialhistory_string == "<N/A>":
                return ""
            socialhistory_list = socialhistory_string.split('; ')
            return socialhistory_list
        return ""

    # Apply the function to the DataFrame and create a new column
    results_df["extracted_socialhistory"] = results_df['SOCIAL HISTORY'].apply(lambda x: extract_socialhistory(x) if x else x)
    df_socialhistory = results_df[['SUBJECT_ID', 'HADM_ID', 'extracted_socialhistory']].explode('extracted_socialhistory').rename(columns={'extracted_socialhistory': 'Social History'})
    # Drop if empty in the Social History column
    df_socialhistory = df_socialhistory[df_socialhistory['Social History'] != '']
    return df_socialhistory

def create_familyhistory_dataframe(results_df):
    ## Create Family history df
    def parse_family_history_data(cell_data):
        if not isinstance(cell_data, str):
            return ""
        # Split the string into individual family history entries
        family_history_entries = re.findall(r"\{.*?\}", cell_data)

        # List to hold the parsed data
        parsed_data = []

        # Extract values for each family history entry
        for entry in family_history_entries:
            # Fuzzy match for 'Family Member'
            family_member_match = re.search(r"'Family[^']*Member': '([^']*)'", entry)
            medical_history_match = re.search(r"'Medical History': '([^']*)'", entry)

            family_member = family_member_match.group(1).replace('<N/A>', '') if family_member_match else ''
            medical_history = medical_history_match.group(1).replace('<N/A>', '') if medical_history_match else ''

            parsed_data.append({
                'Family Member': family_member,
                'Family Medical History': medical_history
            })

        return parsed_data

    # Apply the function to each cell and expand the rows
    expanded_rows = []

    for idx, row in results_df.iterrows():
        if row['FAMILY HISTORY'] != '':  # Check if the family history cell is not empty
            parsed_data = parse_family_history_data(row['FAMILY HISTORY'])
            for item in parsed_data:
                expanded_rows.append({
                    'SUBJECT_ID': row['SUBJECT_ID'],
                    'HADM_ID': row['HADM_ID'],
                    **item
                })

    # Create a new DataFrame with the expanded rows
    df_family_history = pd.DataFrame(expanded_rows)
    ## Check if it is not an empty dataframe
    if not df_family_history.empty:
        df_family_history = df_family_history[df_family_history['Family Medical History'] != '']

    return df_family_history

def data_process_for_db(df_patient, df_vitals, df_history, df_symptoms, df_allergies, df_family_history, df_social_history):
    ## Change column name in df_history from Medical History to Medical_History
    df_history.rename(columns={'Medical History': 'Medical_History'}, inplace=True)
    ## Change column name in df_history from Family edical History to Family_Medical_History
    df_family_history.rename(columns={'Family Medical History': 'Family_Medical_History'}, inplace=True)
    df_family_history.rename(columns={'Family Member': 'Family_Member'}, inplace=True)
    ## Change column name in df_socialhistory from Social History to Social_History
    df_social_history.rename(columns={'Social History': 'Social_History'}, inplace=True)
    df_vitals["VALUE"] = df_vitals["VALUENUM"].astype(str) + " " + df_vitals["VALUEUOM"]
    df_vitals = df_vitals[["SUBJECT_ID", "HADM_ID", "LABEL",  "VALUE"]]
    # Drop AGE_x and change AGE_y to AGE
    df_patient.drop(columns=['AGE_x'], inplace=True)
    df_patient.rename(columns={'AGE_y': 'AGE'}, inplace=True)
    df_patients = df_patient[['SUBJECT_ID','GENDER','AGE','ETHNICITY','RELIGION','MARITAL_STATUS']]
    df_admissions = df_patient[['HADM_ID','SUBJECT_ID','DURATION','ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION','INSURANCE']]
    df_symptoms['Negation'] = df_symptoms['Negation'].fillna('')
    df_symptoms['Duration'] = df_symptoms['Duration'].fillna('')
    df_symptoms['Frequency'] = df_symptoms['Frequency'].fillna('')
    df_symptoms['Intensity'] = df_symptoms['Intensity'].fillna('')
    df_symptoms = df_symptoms.applymap(lambda s: s.lower() if type(s) == str else s)
    # Define the columns to check for empty strings
    check_columns = ['Duration', 'Frequency', 'Intensity', 'Negation']
    # Function to count empty strings in specified columns
    def count_empty_strings(row):
        return row[check_columns].apply(lambda x: x == '').sum()
    # Group by the specified columns and keep the row with the fewest empty strings
    df_symptoms['empty_count'] = df_symptoms.apply(count_empty_strings, axis=1)
    df_symptoms = df_symptoms.loc[df_symptoms.groupby(['SUBJECT_ID', 'HADM_ID', 'Symptom'])['empty_count'].idxmin()]
    # Drop the helper column
    df_symptoms = df_symptoms.drop(columns=['empty_count'])
    # Convert to string 
    df_patients['SUBJECT_ID'] = df_patients['SUBJECT_ID'].astype(str)
    
    df_admissions['SUBJECT_ID'] = df_admissions['SUBJECT_ID'].astype(str)
    df_admissions['HADM_ID'] = df_admissions['HADM_ID'].astype(str)

    df_vitals['SUBJECT_ID'] = df_vitals['SUBJECT_ID'].astype(str)
    df_vitals['HADM_ID'] = df_vitals['HADM_ID'].astype(str)

    df_history['SUBJECT_ID'] = df_history['SUBJECT_ID'].astype(str)
    df_history['HADM_ID'] = df_history['HADM_ID'].astype(str)

    df_symptoms['SUBJECT_ID'] = df_symptoms['SUBJECT_ID'].astype(str)
    df_symptoms['HADM_ID'] = df_symptoms['HADM_ID'].astype(str)

    df_allergies['SUBJECT_ID'] = df_allergies['SUBJECT_ID'].astype(str)
    df_allergies['HADM_ID'] = df_allergies['HADM_ID'].astype(str)

    df_family_history['SUBJECT_ID'] = df_family_history['SUBJECT_ID'].astype(str)
    df_family_history['HADM_ID'] = df_family_history['HADM_ID'].astype(str)

    df_social_history['SUBJECT_ID'] = df_social_history['SUBJECT_ID'].astype(str)
    df_social_history['HADM_ID'] = df_social_history['HADM_ID'].astype(str)
    
    # Final Save 
    return df_patients, df_admissions, df_vitals, df_history, df_symptoms, df_allergies, df_family_history, df_social_history