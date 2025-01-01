import os
import re
import pandas as pd
from openai import OpenAI
from datetime import datetime

from data_cleaning_coral.data_cleaning_function.create_dataframe import *
from data_cleaning_coral.data_cleaning_function.prompts import *
from config import config

class MedicalDataProcessor:
    def __init__(self, llm):
        """
        Initialize the processor using configuration from the config file.
        :param secret_file: Path to the file containing the OpenAI API key.
        """
        self.llm = llm
        self.final_data = {}
        self.raw_data_paths = config["raw_data_paths"]

    def read_txt_files(self):
        """
        Iteratively reads all txt files in the provided directories and creates a DataFrame with two columns:
        'id' (from filename) and 'text' (content of the file).
        """
        data = []

        for folder_path in self.raw_data_paths:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        file_id = os.path.splitext(file_name)[0]
                        data.append({'id': file_id, 'text': text})

        df = pd.DataFrame(data)
        return df
    
    def process_data(self, df):
        """
        Process each row from the data using GPT-based prompts and store the results in a DataFrame.
        :param df: DataFrame containing the raw text data.
        :return: A DataFrame containing the extracted symptoms, medical history, allergies, etc.
        """
        # Initialize the results DataFrame with required columns
        results_df = pd.DataFrame(columns=[
            'SUBJECT_ID', 'HADM_ID', 'SYMPTOMS', 'MEDICAL HISTORY', 'ALLERGIES', 'SOCIAL HISTORY', 'FAMILY HISTORY'
        ])
        # Iterate through the df DataFrame
        for index, row in df.iterrows():
            # Set initial empty values for extracted fields
            additional_symptoms = ""
            medical_history = ""
            allergies = ""
            social_history = ""
            family_history = ""

            # Extract fields
            SUBJECT_ID = row["id"]
            HADM_ID = row['id']
            history_text = row["text"]
            medical_history_text = row["text"]
            allergies_text = row["text"]
            social_history_text = row["text"]
            family_history_text = row["text"]

            # Generate prompts and run GPT for each category
            symptom_prompt = extract_symptom_prompt(history_text)
            additional_symptoms = self.llm.run_gpt(symptom_prompt)

            history_prompt = format_history_prompt(medical_history_text)
            medical_history = self.llm.run_gpt(history_prompt)

            allergies_prompt = extract_allergies(allergies_text)
            allergies = self.llm.run_gpt(allergies_prompt)

            social_history_prompt = extract_socialhistory(social_history_text)
            social_history = self.llm.run_gpt(social_history_prompt)

            family_history_prompt = extract_familyhistory(family_history_text)
            family_history = self.llm.run_gpt(family_history_prompt)

            # Append the extracted data to the results DataFrame
            results_df = results_df._append({
                'SUBJECT_ID': SUBJECT_ID,
                'HADM_ID': HADM_ID,
                'SYMPTOMS': additional_symptoms,
                'MEDICAL HISTORY': medical_history,
                'ALLERGIES': allergies,
                'SOCIAL HISTORY': social_history,
                'FAMILY HISTORY': family_history
            }, ignore_index=True)

        return results_df
    
    def clean_patient_info(self):
        df = pd.read_csv("/Users/huiziyu/Dropbox/AIPatient_OOD/data/raw/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated/subject-info.csv")
        df = df.rename(columns={
            'coral_idx': 'SUBJECT_ID',
            'Sex': 'GENDER',
            'BirthDate': 'BirthDate',
            'UCSFDerivedRaceEthnicity_X': 'ETHNICITY'
        })
        
        df['SUBJECT_ID'] = df['SUBJECT_ID'].astype(str)
        # Convert BirthDate to datetime
        df['BirthDate'] = pd.to_datetime(df['BirthDate'], errors='coerce')
        df["HADM_ID"] = df["SUBJECT_ID"]
        # Get today's date
        today = datetime.today()

        # Calculate Age based on BirthDate and today's date
        df['AGE'] = df['BirthDate'].apply(lambda birth_date: today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day)) if pd.notnull(birth_date) else None)

        return df[["SUBJECT_ID", "HADM_ID", "GENDER", "AGE", "ETHNICITY"]]
    
    def clean_admission_info(self):
        """Create Admission with HADM_ID from 0 to 39, assign ADMISSION_TYPE for pancreatic and breast cancer."""
        # Generate a list of HADM_IDs from 0 to 39
        subject_ids  = list(range(40))
        hadm_ids = list(range(40))
        # Assign admission types: first 20 for pancreatic cancer, rest for breast cancer
        admission_types = ['Pancreatic Cancer' if i < 20 else 'Breast Cancer' for i in hadm_ids]
        # Create a DataFrame with the HADM_ID and ADMISSION_TYPE
        admission_df = pd.DataFrame({
            'SUBJECT_ID': subject_ids,
            'HADM_ID': hadm_ids,
            'ADMISSION_TYPE': admission_types
        })
        admission_df['SUBJECT_ID'] = admission_df['SUBJECT_ID'].astype(str)
        admission_df['HADM_ID'] = admission_df['HADM_ID'].astype(str)
        # Optionally, display or return the DataFrame for further processing
        return admission_df
        
    
    def orchestrator(self):
        """
        Orchestrates the data processing by reading the raw data, applying GPT prompts, and generating a results DataFrame.
        """
        # Step 1: Read all text files into a single dataframe
        raw_df = self.read_txt_files()
        # Step 2: Extract Relevant info 
        processed_df = self.process_data(raw_df[:40])
        df_patient = self.clean_patient_info()
        df_admission = self.clean_admission_info()
        # Step 3: Create entity specific dataframe 
        df_symptoms = create_symptom_dataframe(processed_df)
        df_history = create_history_dataframe(processed_df)
        df_allergies = create_allergies_dataframe(processed_df)
        df_social_history = create_socialhistory_dataframe(processed_df)
        df_family_history = create_familyhistory_dataframe(processed_df)
        
        # Additional data cleaning
        ## Change column name in df_history from Medical History to Medical_History
        df_history.rename(columns={'Medical History': 'Medical_History'}, inplace=True)
        ## Change column name in df_history from Family edical History to Family_Medical_History
        df_family_history.rename(columns={'Family Medical History': 'Family_Medical_History'}, inplace=True)
        df_family_history.rename(columns={'Family Member': 'Family_Member'}, inplace=True)
        ## Change column name in df_socialhistory from Social History to Social_History
        df_social_history.rename(columns={'Social History': 'Social_History'}, inplace=True)
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
        
        # Store final data
        self.final_data['patient'] = df_patient
        self.final_data['admission'] = df_admission
        self.final_data['symptoms'] = df_symptoms
        self.final_data['history'] = df_history
        self.final_data['allergies'] = df_allergies
        self.final_data['family_history'] = df_family_history
        self.final_data['social_history'] = df_social_history
        
        # Finally saving the data
        df_admission.to_csv(f"{config['data_path']}/df_admission.csv")
        df_patient.to_csv(f"{config['data_path']}/df_patients.csv")
        df_symptoms.to_csv(f"{config['data_path']}/df_symptoms.csv")
        df_history.to_csv(f"{config['data_path']}/df_history.csv")
        df_allergies.to_csv(f"{config['data_path']}/df_allergies.csv")
        df_family_history.to_csv(f"{config['data_path']}/df_family_history.csv")
        df_social_history.to_csv(f"{config['data_path']}/df_social_history.csv")