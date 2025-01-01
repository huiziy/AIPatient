from graph_construction.graph_construction_function.entity_creation import *
from neo4j import GraphDatabase
import pandas as pd

class Neo4jDatabase:
    def __init__(self, uri, user, password):
        """Initialize the Neo4j driver"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Input data
        self.nodes = [
            {'labels': 'Patient', 'properties': ['SUBJECT_ID', 'GENDER', 'AGE', 'ETHNICITY']},
            {'labels': 'Admission', 'properties': ['HADM_ID', 'ADMISSION_TYPE']},
            {'labels': 'Symptom', 'properties': ['name']},
            {'labels': 'Duration', 'properties': ['name']},
            {'labels': 'Intensity', 'properties': ['name']},
            {'labels': 'Frequency', 'properties': ['name']},
            {'labels': 'History', 'properties': ['name']},
            {'labels': 'Allergy', 'properties': ['name']},
            {'labels': 'SocialHistory', 'properties': ['description']},
            {'labels': 'FamilyMember', 'properties': ['name']},
            {'labels': 'FamilyMedicalHistory', 'properties': ['name']}
        ]

        self.relationships = [
            {'relationship': 'HAS_ADMISSION', 'source': 'Patient', 'target': ['Admission']},
            {'relationship': 'HAS_MEDICAL_HISTORY', 'source': 'Patient', 'target': ['History']},
            {'relationship': 'HAS_FAMILY_MEMBER', 'source': 'Patient', 'target': ['FamilyMember']},
            {'relationship': 'HAS_SYMPTOM', 'source': 'Admission', 'target': ['Symptom']},
            {'relationship': 'HAS_SOCIAL_HISTORY', 'source': 'Admission', 'target': ['SocialHistory']},
            {'relationship': 'HAS_ALLERGY', 'source': 'Admission', 'target': ['Allergy']},
            {'relationship': 'HAS_NOSYMPTOM', 'source': 'Admission', 'target': ['Symptom']},
            {'relationship': 'HAS_DURATION', 'source': 'Symptom', 'target': ['Duration']},
            {'relationship': 'HAS_INTENSITY', 'source': 'Symptom', 'target': ['Intensity']},
            {'relationship': 'HAS_FREQUENCY', 'source': 'Symptom', 'target': ['Frequency']},
            {'relationship': 'HAS_MEDICAL_HISTORY', 'source': 'FamilyMember', 'target': ['FamilyMedicalHistory']}
        ]
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def load_patients(self, df_patients):
        """Load patients into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_patients.iterrows():
                session.execute_write(create_patient, row['SUBJECT_ID'], row['GENDER'], row['AGE'], row['ETHNICITY'])
                
    def load_admission(self, df_admission):
        """Load patients into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_admission.iterrows():
                session.execute_write(create_admission, row['SUBJECT_ID'], row['HADM_ID'], row['ADMISSION_TYPE'])
        
    def load_symptoms(self, df_symptoms):
        """Load symptoms into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_symptoms.iterrows():
                session.execute_write(
                    create_symptom,
                    row['HADM_ID'],
                    row['Symptom'],
                    row['Duration'],
                    row['Frequency'],
                    row['Intensity'],
                    row['Negation']
                )
    
    def load_history(self, df_history):
        """Load medical history into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_history.iterrows():
                if pd.notna(row['Medical_History']):
                    session.execute_write(create_history, row['SUBJECT_ID'], row['Medical_History'])
    
    def load_allergies(self, df_allergies):
        """Load allergies into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_allergies.iterrows():
                session.execute_write(create_allergy, row['SUBJECT_ID'], row['HADM_ID'], row['Allergies'])
    
    def load_social_history(self, df_social_history):
        """Load social history into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_social_history.iterrows():
                if pd.notna(row['Social_History']):
                    session.execute_write(create_social_history, row['SUBJECT_ID'], row['HADM_ID'], row['Social_History'])
    
    def load_family_history(self, df_family_history):
        """Load family history into Neo4j"""
        with self.driver.session() as session:
            for index, row in df_family_history.iterrows():
                if pd.notna(row['Family_Medical_History']) and pd.notna(row['Family_Member']):
                    session.execute_write(create_family_history, row['SUBJECT_ID'], row['Family_Member'], row['Family_Medical_History'])
    
    def load_all_data(self, df_patients, df_admission, df_symptoms, df_history, df_allergies, df_social_history, df_family_history):
        """Load all data into Neo4j"""
        self.load_patients(df_patients)
        self.load_admission(df_admission)
        self.load_symptoms(df_symptoms)
        self.load_history(df_history)
        self.load_allergies(df_allergies)
        self.load_social_history(df_social_history)
        self.load_family_history(df_family_history)


    def db_creation_orchestrator(self, df_patients, df_admission, df_symptoms, df_history, df_allergies, df_social_history, df_family_history):
        """Orchestrates the data loading process into Neo4j"""
        try:
            # Step 1: Clear existing data in the database
            print("Clearing the Neo4j database...")
            self.clear_database()

            # Step 2: Load data into Neo4j
            print("Loading data into Neo4j...")
            self.load_all_data(df_patients, df_admission, df_symptoms, df_history, df_allergies, df_social_history, df_family_history)

        except Exception as e:
            print(f"An error occurred during the data loading process: {e}")
        
        finally:
            # Step 3: Close the connection to Neo4j
            print("Closing the Neo4j connection...")
            self.close()
            print("Data loading process completed.")
            
            
    
    def get_random_patient_admission(self):
        """Fetch a random patient and admission"""
        with self.driver.session() as session:
            result = session.execute_read(self._fetch_random_patient_admission)
            return result

    @staticmethod
    def _fetch_random_patient_admission(tx):
        """Fetch a random patient admission from the database"""
        query = """
        MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission)
        WITH p, a, rand() AS random
        ORDER BY random
        LIMIT 1
        RETURN p.SUBJECT_ID AS SubjectID, a.HADM_ID AS AdmissionID
        """
        result = tx.run(query)
        return result.single()

    def execute_cypher_query(self, cypher_query):
        """Execute a Cypher query"""
        with self.driver.session() as session:
            result = session.execute_read(self._run_cypher_query, cypher_query)
            return result

    @staticmethod
    def _run_cypher_query(tx, cypher_query):
        """Run a Cypher query and return the results as a list of dictionaries"""
        result = tx.run(cypher_query)
        return [record.data() for record in result]

    # New methods added based on your provided functions

    def fetch_all_symptoms(self, hadm_id):
        """Fetch all symptoms for a given admission ID"""
        query = f"""
            MATCH (a:Admission {{HADM_ID: {hadm_id}}})-[r:HAS_SYMPTOM]->(s:Symptom)
            RETURN a, r, s
        """
        retrieved = self.execute_cypher_query(query)
        symptoms = [entry['s']['name'] for entry in retrieved]
        return query, symptoms

    def fetch_all_medicalhistory(self, hadm_id):
        """Fetch all medical history for a given admission ID"""
        query = f"""
            MATCH (p:Patient)-[r:HAS_MEDICAL_HISTORY]->(h:History)
            WHERE EXISTS((p)-[:HAS_ADMISSION]->(:Admission {{HADM_ID: {hadm_id}}}))
            RETURN p, r, h
        """
        retrieved = self.execute_cypher_query(query)
        history = [entry['h']['name'] for entry in retrieved]
        return query, history

    def fetch_all_allergies(self, hadm_id):
        """Fetch all allergies for a given admission ID"""
        query = f"""
            MATCH (a:Admission {{HADM_ID: {hadm_id}}})-[r:HAS_ALLERGY]->(al:Allergy)
            MATCH (p:Patient)-[:HAS_ADMISSION]->(a)
            RETURN p, a, r, al
        """
        retrieved = self.execute_cypher_query(query)
        allergies = [entry['al']['name'] for entry in retrieved]
        return query, allergies
    
    
    ## Reformat Schema
    def reformat_schema(self, nodes, relationships):
        # Format node properties
        node_properties = "### Node Properties\n"
        for node in nodes:
            label = node['labels']
            properties = ", ".join(node['properties'])
            node_properties += f"- {label}: {properties}\n"

        # Format relationships
        relationship_info = "### Relationships\n"
        for rel in relationships:
            relationship = rel['relationship']
            source = rel['source']
            target = ", ".join(rel['target'])
            relationship_info += f"- {source} -> {target}: {relationship}\n"

        return node_properties + "\n" + relationship_info

    

    # Call the function and print the result
    def generate_schema(self):
        schema_full = self.reformat_schema(self.nodes, self.relationships)
        return schema_full