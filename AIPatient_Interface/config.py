config = {
    "raw_data_paths": [ # Replace with path to Raw MIMIC-III data
        "MIMICIII/mimic-iii-clinical-database-1.4/Raw"
    ],
    "model": "gpt-4-turbo",
    "max_tokens": 4096,
    "temperature": 0,
    "secret_file":"", # Replace with path to secret file
    "data_path": "", # Replace with path for storing data
    "db_uri": "", # Replace with your local host or Aura DB uri
    "db_user": "neo4j", # Replace with your user name 
    "db_password": "AIPatient2024" # Replace with your password
}