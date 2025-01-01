config = {
    "raw_data_paths": [ # Replace with the path to two CORAL datasets
        "coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated/breastca/",
        "coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated/pdac/"
    ],
    "model": "gpt-4-turbo",
    "max_tokens": 4096,
    "temperature": 0,
    "secret_file":"", # Replace with path to secret file
    "data_path": "", # Replace with path for storing data
    "patient_info_file": "coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/annotated/subject-info.csv", # Replace with location to patient Info File
    "db_uri": "", # Replace with your local host or Aura DB uri
    "db_user": "neo4j", # Replace with your user name 
    "db_password": "AIPatient2024" # Replace with your password
}