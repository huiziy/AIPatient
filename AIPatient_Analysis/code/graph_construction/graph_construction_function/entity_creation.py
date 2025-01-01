
def create_patient(tx, subject_id, gender, age, ethnicity, religion, marital_status):
    tx.run("""
        MERGE (p:Patient {SUBJECT_ID: $subject_id})
        SET p.GENDER = $gender, p.AGE = $age, p.ETHNICITY = $ethnicity, p.RELIGION = $religion, p.MARITAL_STATUS = $marital_status
    """, subject_id=subject_id, gender=gender, age=age, ethnicity=ethnicity, religion=religion, marital_status=marital_status)

def create_admission(tx, hadm_id, subject_id, duration, admission_type, admission_location, discharge_location, insurance):
    tx.run("""
        MERGE (a:Admission {HADM_ID: $hadm_id})
        SET a.DURATION = $duration, a.ADMISSION_TYPE = $admission_type, a.ADMISSION_LOCATION = $admission_location, a.DISCHARGE_LOCATION = $discharge_location, a.INSURANCE = $insurance
        WITH a
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        MERGE (p)-[:HAS_ADMISSION]->(a)
    """, hadm_id=hadm_id, subject_id=subject_id, duration=duration, admission_type=admission_type, admission_location=admission_location, discharge_location=discharge_location, insurance=insurance)

def create_symptom(tx, hadm_id, symptom, duration, frequency, intensity, negation):
    if '[n]' in negation:
        clean_symptom = symptom.strip()
        query = """
            MERGE (s:Symptom {name: $clean_symptom})
            WITH s
            MATCH (a:Admission {HADM_ID: $hadm_id})
            MERGE (a)-[:HAS_NOSYMPTOM]->(s)
            FOREACH (d IN CASE WHEN $duration <> '' THEN [1] ELSE [] END |
                MERGE (dur:Duration {name: $duration})
                MERGE (s)-[:HAS_DURATION]->(dur)
            )
            FOREACH (f IN CASE WHEN $frequency <> '' THEN [1] ELSE [] END |
                MERGE (freq:Frequency {name: $frequency})
                MERGE (s)-[:HAS_FREQUENCY]->(freq)
            )
            FOREACH (i IN CASE WHEN $intensity <> '' THEN [1] ELSE [] END |
                MERGE (int:Intensity {name: $intensity})
                MERGE (s)-[:HAS_INTENSITY]->(int)
            )
        """
        tx.run(query, hadm_id=hadm_id, clean_symptom=clean_symptom, duration=duration, frequency=frequency, intensity=intensity)
    else:
        query = """
            MERGE (s:Symptom {name: $symptom})
            WITH s
            MATCH (a:Admission {HADM_ID: $hadm_id})
            MERGE (a)-[:HAS_SYMPTOM]->(s)
            FOREACH (d IN CASE WHEN $duration <> '' THEN [1] ELSE [] END |
                MERGE (dur:Duration {name: $duration})
                MERGE (s)-[:HAS_DURATION]->(dur)
            )
            FOREACH (f IN CASE WHEN $frequency <> '' THEN [1] ELSE [] END |
                MERGE (freq:Frequency {name: $frequency})
                MERGE (s)-[:HAS_FREQUENCY]->(freq)
            )
            FOREACH (i IN CASE WHEN $intensity <> '' THEN [1] ELSE [] END |
                MERGE (int:Intensity {name: $intensity})
                MERGE (s)-[:HAS_INTENSITY]->(int)
            )
        """
        tx.run(query, hadm_id=hadm_id, symptom=symptom, duration=duration, frequency=frequency, intensity=intensity)

def create_history(tx, subject_id, history):
    tx.run("""
        MERGE (h:History {name: $history})
        WITH h
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        MERGE (p)-[:HAS_MEDICAL_HISTORY]->(h)
    """, subject_id=subject_id, history=history)

def create_vital(tx, subject_id, hadm_id, label, value):
    tx.run("""
        MERGE (v:Vital {LABEL: $label, VALUE: $value})
        WITH v
        MATCH (a:Admission {HADM_ID: $hadm_id})
        MERGE (a)-[:HAS_VITAL]->(v)
    """, subject_id=subject_id, hadm_id=hadm_id, label=label, value=value)


def create_allergy(tx, subject_id, hadm_id, allergy):
    tx.run("""
        MERGE (al:Allergy {name: $allergy})
        WITH al
        MATCH (a:Admission {HADM_ID: $hadm_id})
        MERGE (a)-[:HAS_ALLERGY]->(al)
        WITH a
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        MERGE (p)-[:HAS_ADMISSION]->(a)
    """, subject_id=subject_id, hadm_id=hadm_id, allergy=allergy)

def create_social_history(tx, subject_id, hadm_id, social_history):
    tx.run("""
        MERGE (sh:SocialHistory {description: $social_history})
        WITH sh
        MATCH (a:Admission {HADM_ID: $hadm_id})
        MERGE (a)-[:HAS_SOCIAL_HISTORY]->(sh)
        WITH a
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        MERGE (p)-[:HAS_ADMISSION]->(a)
    """, subject_id=subject_id, hadm_id=hadm_id, social_history=social_history)

def create_family_history(tx, subject_id, family_member, family_medical_history):
    tx.run("""
        MERGE (fm:FamilyMember {name: $family_member})
        WITH fm
        MATCH (p:Patient {SUBJECT_ID: $subject_id})
        MERGE (p)-[:HAS_FAMILY_MEMBER]->(fm)
        WITH fm
        MERGE (fmh:FamilyMedicalHistory {name: $family_medical_history})
        MERGE (fm)-[:HAS_MEDICAL_HISTORY]->(fmh)
    """, subject_id=subject_id, family_member=family_member, family_medical_history=family_medical_history)