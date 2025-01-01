import re
import pandas as pd
from data_cleaning_coral.data_cleaning_function.prompts import *

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