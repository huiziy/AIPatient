

import os
import re
import pandas as pd
from datetime import datetime

from data_cleaning_mimic.data_cleaning_function.create_dataframe import *
from data_cleaning_mimic.data_cleaning_function.prompts import *
from config import config

class MedicalDataProcessor:
    def __init__(self, llm, total_cases):
        """
        Initialize the processor using configuration from the config file.
        :param secret_file: Path to the file containing the OpenAI API key.
        """
        self.llm = llm
        self.total_cases = total_cases
        self.final_data = {}
        self.raw_data_paths = config["raw_data_paths"]
        self.clean_data_path = config["data_path"]
        # ICD category: https://web.archive.org/web/20140611114252/http://simba.isr.umich.edu/restricted/docs/Mortality/icd_09_codes.pdf
        self.icd9_categories = [
            {"range_start": 1, "range_end": 139, "category": "infectious and parasitic diseases"},
            {"range_start": 140, "range_end": 239, "category": "neoplasms"},
            {"range_start": 240, "range_end": 279, "category": "endocrine, nutritional and metabolic diseases, and immunity disorders"},
            {"range_start": 280, "range_end": 289, "category": "diseases of the blood and blood-forming organs"},
            {"range_start": 290, "range_end": 319, "category": "mental disorders"},
            {"range_start": 320, "range_end": 389, "category": "diseases of the nervous system and sense organs"},
            {"range_start": 390, "range_end": 459, "category": "diseases of the circulatory system"},
            {"range_start": 460, "range_end": 519, "category": "diseases of the respiratory system"},
            {"range_start": 520, "range_end": 579, "category": "diseases of the digestive system"},
            {"range_start": 580, "range_end": 629, "category": "diseases of the genitourinary system"},
            {"range_start": 630, "range_end": 679, "category": "complications of pregnancy, childbirth, and the puerperium"},
            {"range_start": 680, "range_end": 709, "category": "diseases of the skin and subcutaneous tissue"},
            {"range_start": 710, "range_end": 739, "category": "diseases of the musculoskeletal system and connective tissue"},
            {"range_start": 740, "range_end": 759, "category": "congenital anomalies"},
            {"range_start": 760, "range_end": 779, "category": "certain conditions originating in the perinatal period"},
            {"range_start": 780, "range_end": 799, "category": "symptoms, signs, and ill-defined conditions"},
            {"range_start": 800, "range_end": 999, "category": "injury and poisoning"},
            {"range_start": 0, "range_end": 0, "category": "external causes of injury and supplemental classification"},
        ]
        
    def data_cleaning_orchestrator(self):
        print("Cleaning Datasets")
        self.df_adm_raw = pd.read_csv(self.raw_data_paths[0] + "/ADMISSIONS.csv")
        self.df_patients_raw = pd.read_csv(self.raw_data_paths[0] + "/PATIENTS.csv")
        self.diagnosis = pd.read_csv(self.raw_data_paths[0] + "/DIAGNOSES_ICD.csv")
        self.icd = pd.read_csv(self.raw_data_paths[0] + "/D_ICD_DIAGNOSES.csv")
        self.items = pd.read_csv(self.raw_data_paths[0] + "/D_ITEMS.csv")
        self.notes_raw = pd.read_csv(self.raw_data_paths[0] + "/NOTEEVENTS.csv")
        # Create df_adm 
        self.df_adm = clean_adm(self.df_adm_raw, self.df_patients_raw)
        # Create patient selected
        self.patient_selected = patient_selection(self.diagnosis, self.icd, self.df_adm, self.icd9_categories, self.total_cases)
        # Create chart_event_data
        self.df_chart_raw = load_filtered_csv_dask(self.raw_data_paths[0] + "/CHARTEVENTS.csv", self.patient_selected)
        self.df_chart = clean_vitals(self.df_chart_raw, self.items)
        # Create Notes data 
        self.df_notes = pd.merge(self.notes_raw, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_discharge_notes = clean_notes(self.df_notes)
        # Apply the extraction function to each cell in the DataFrame
        df_extracted = self.df_discharge_notes["TEXT"].apply(extract_sections_from_summary)
        # Combine the extracted sections with the original DataFrame
        self.df_combined = pd.concat([self.df_discharge_notes, df_extracted], axis=1)
        # keep only SUBJECT_ID, HADM_ID, CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS, and PAST MEDICAL HISTORY
        self.df_combined = self.df_combined[['SUBJECT_ID', 'HADM_ID', 'CHIEF COMPLAINT', 'HISTORY OF PRESENT ILLNESS', 'PAST MEDICAL HISTORY', 'REVIEW OF SYSTEM', 'ALLERGIES', 'SOCIAL HISTORY', 'FAMILY HISTORY']]
        # Keep if only any of CHIEF COMPLAINT or HISTORY OF PRESENT ILLNESS is not empty
        self.df_combined = self.df_combined[self.df_combined['CHIEF COMPLAINT'].notnull() | self.df_combined['HISTORY OF PRESENT ILLNESS'].notnull()]
        
        print("NER from Discharge Summary")
        # Extract info from Notes 
        ## Additional symptom extraction
        results_df = pd.DataFrame(columns=['SUBJECT_ID', 'HADM_ID',"SYMPTOMS", "MEDICAL HISTORY", "ALLERGIES", "SOCIAL HISTORY", "FAMILY HISTORY"])
        counts = 0
        # Iterate through the df_combined DataFrame
        for index, row in self.df_combined.iterrows():
            print(f"Processing row {index}...")
            print(counts)
            counts += 1
            ## Set to Empty
            additional_symptoms = ""
            medical_history = ""
            allergies = ""
            social_history = ""
            family_history = ""

            ## Extract fields
            SUBJECT_ID = row["SUBJECT_ID"]
            HADM_ID = row["HADM_ID"]
            history_text = row["HISTORY OF PRESENT ILLNESS"]
            medical_history_text = row["PAST MEDICAL HISTORY"]
            allergies_text = row["ALLERGIES"]
            social_history_text = row["SOCIAL HISTORY"]
            family_history_text = row["FAMILY HISTORY"]

            ## Add Review of System if any
            if row["REVIEW OF SYSTEM"]:
                history_text = f"The review of system of the patient is {row['REVIEW OF SYSTEM']}. {history_text}"

            ## Add chief complaint if any
            if row["CHIEF COMPLAINT"]:
                history_text = f"The chief complaint of the patient is {row['CHIEF COMPLAINT']}. {history_text}"

            # Extract symptoms prompt
            symptom_prompt = extract_symptom_prompt(history_text)
            additional_symptoms = self.llm.run_gpt(symptom_prompt)

            # Extract medical history prompt
            if row["PAST MEDICAL HISTORY"]:
                history_prompt = format_history_prompt(medical_history_text)
            medical_history = self.llm.run_gpt(history_prompt)

            # Extract allergies prompt
            if row["ALLERGIES"]:
                allergies_prompt = extract_allergies(allergies_text)
                allergies = self.llm.run_gpt(allergies_prompt)

            # Extract social history text
            if row["SOCIAL HISTORY"]:
                social_history_prompt = extract_socialhistory(social_history_text)
                social_history = self.llm.run_gpt(social_history_prompt)

            # Extract family history text
            if row["FAMILY HISTORY"]:
                family_history_prompt = extract_familyhistory(family_history_text)
                family_history = self.llm.run_gpt(family_history_prompt)
            # Append the results to the results DataFrame
            results_df = results_df._append({
                'SUBJECT_ID': SUBJECT_ID,
                'HADM_ID': HADM_ID,
                'SYMPTOMS': additional_symptoms,
                'MEDICAL HISTORY': medical_history,
                'ALLERGIES': allergies,
                'SOCIAL HISTORY': social_history,
                'FAMILY HISTORY': family_history
            }, ignore_index=True)
            self.results_df = results_df
        
        print("Export Datasets")    
        ## Save to local files 
        self.df_symptom = create_symptom_dataframe(self.results_df)
        self.df_history = create_history_dataframe(self.results_df)
        self.df_allergies = create_allergies_dataframe(self.results_df)
        self.df_family_history = create_familyhistory_dataframe(self.results_df)
        self.df_socialhistory = create_socialhistory_dataframe(self.results_df)
        
        
        ## Export to Construct Graph
        self.df_adm = pd.merge(self.df_adm, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_chart = pd.merge(self.df_chart, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_history = pd.merge(self.df_history, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_symptom = pd.merge(self.df_symptom, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_allergies = pd.merge(self.df_allergies, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_family_history = pd.merge(self.df_family_history, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        self.df_socialhistory = pd.merge(self.df_socialhistory, self.patient_selected, on=['SUBJECT_ID', 'HADM_ID'])
        
        ## Second round of cleaning for data base 
        self.df_patient, self.df_adm, self.df_chart, self.df_history, self.df_symptom, self.df_allergies, self.df_family_history,  self.df_socialhistory = data_process_for_db(self.df_adm, self.df_chart, self.df_history, self.df_symptom, self.df_allergies, self.df_family_history,  self.df_socialhistory)
        
        self.df_patient.to_csv(self.clean_data_path + "/patients.csv", index=False)
        self.df_adm.to_csv(self.clean_data_path + "/admissions.csv", index=False)
        self.df_chart.to_csv(self.clean_data_path + "/vitals.csv", index=False)
        self.df_history.to_csv(self.clean_data_path + "/history.csv", index=False)
        self.df_symptom.to_csv(self.clean_data_path + "/symptoms.csv", index=False)
        self.df_allergies.to_csv(self.clean_data_path + "/allergies.csv", index=False)
        self.df_family_history.to_csv(self.clean_data_path + "/family_history.csv", index=False)
        self.df_socialhistory.to_csv(self.clean_data_path + "/social_history.csv", index=False)
