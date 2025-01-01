import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time
from neo4j import GraphDatabase
from streamlit_chat import message
from py2neo import Graph, Node as Py2neoNode, Relationship as Py2neoRelationship
from streamlit_agraph import agraph, Node, Edge, Config
from agents.agents_class import Agents
from Neo4jDatabase.Neo4jDatabase_class import Neo4jDatabase
from Neo4jDatabase.Neo4jDatabase_visualizer import Neo4jGraphVisualizer
from llm_models.llm_model_class import LLM_Models
import random
import logging
import os
import itertools
from config import config
import re

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('interactive_session.log'), logging.StreamHandler()])

## Generate personality profiles
big_five_traits = {
            "Openness": ["Practical, conventional, prefers routine", "Curious, wide range of interests, independent"],
            "Conscientiousness": ["Impulsive, careless, disorganized", "Hardworking, dependable, organized"],
            "Extraversion": ["Quiet, reserved, withdrawn", "Outgoing, warm, seeks adventure"],
            "Agreeableness": ["Critical, uncooperative, suspicious", "Helpful, trusting, empathetic"],
            "Neuroticism": ["Calm, even-tempered, secure", "Anxious, unhappy, prone to negative emotions"]
            }
# Generate all combinations
all_combinations = list(itertools.product(*big_five_traits.values()))

# Convert to list of dictionaries to represent each personality type
list_of_personalities = []
for combination in all_combinations:
    personality = dict(zip(big_five_traits.keys(), combination))
    list_of_personalities.append(personality)

def main():
    # Initialize the session state for holding the conversation if it doesn't exist
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Load necessary credentials and set up the database connection
    model_type = "gpt"
    verbose = True
    
    ## Initiative db
    if 'db' not in st.session_state:
        uri = config["db_uri"]
        user = config["db_user"]
        password = config["db_password"]
        st.session_state.db = Neo4jDatabase(uri, user, password)
        ## Initiative llm 
        st.session_state.llm_model = LLM_Models()
        ## Initiative agents
        st.session_state.agents = Agents(st.session_state.db, st.session_state.llm_model, model_type)
    
    # Check if a patient admission is already stored in session state, if not, get a new one
    if 'patient_admission' not in st.session_state:
        patient_admission = st.session_state.db.get_random_patient_admission()
        st.session_state.patient_admission = patient_admission
        st.session_state.graph_generated = False
        if verbose:
            selected_personality = random.choice(list_of_personalities)  
            st.session_state.personality_profile = list(selected_personality.values())  
            st.session_state.personality_profile.append("Verbose")  
        else:
            st.session_state.personality_profile = ["Responsible", "Organized", "Analytical", "Terse"]
        st.session_state.model = model_type
        st.session_state.conversation_history = f"The patient has ID {patient_admission['SubjectID']}, and the admission ID {patient_admission['AdmissionID']}"
        logging.info(f"New patient drawn: {patient_admission['SubjectID']}")

    patient_admission = st.session_state.patient_admission
    personality_profile = st.session_state.personality_profile

    # Generate the graph if it hasn't been generated yet
    if not st.session_state.graph_generated:
        visualizer = Neo4jGraphVisualizer(config["db_uri"], config["db_user"], config["db_password"])
        results = visualizer.fetch_data(int(patient_admission['AdmissionID']))
        nodes, edges = visualizer.create_nodes_edges(results)
        st.session_state.nodes = nodes
        st.session_state.edges = edges
        st.session_state.graph_generated = True
        logging.info("Graph generated and stored in session state.")

    # Create two columns for QA and Knowledge Graph
    col1, col2 = st.columns(2)

    with col1:
        st.header("QA Interaction")
        # Display the previous conversations
        chat_placeholder = st.empty()
        st.session_state.agents.display_conversation(chat_placeholder)

        # Input for new queries from the doctor
        user_input = st.text_input("Enter your query as a doctor:", key="doctor_query")

        # Submit button to process the query
        if st.button("Submit"):
            if user_input.lower() == 'exit':
                st.write("Terminating session.")
                st.session_state.db.close()
                return

            # Process the interaction
            patient_response, updated_conversation_history = st.session_state.agents.interactive_session(user_input, st.session_state.conversation_history, patient_admission, personality_profile)
            # Update the chat display and the session state
            st.session_state.conversation.append((user_input, patient_response))
            st.session_state.conversation_history = updated_conversation_history
            # Refresh the chat display to include the new conversation
            st.session_state.agents.display_conversation(chat_placeholder)
        
    with col2:
        st.header("Session Info")
        st.markdown("#### Large Language Model")
        st.write("Claude 3.5 Sonnet")
        st.markdown("#### Patient Personality")
        st.write(", ".join(st.session_state.personality_profile))
        st.header("Converstion Summary")
        st.write(st.session_state.conversation_history)
        st.header("Electronic Health Records Knowledge Graph")
        # Fetch and visualize the knowledge graph only if it hasn't been generated yet
        visualizer = Neo4jGraphVisualizer(config["db_uri"], config["db_user"], config["db_password"])
        visualizer.visualize_graph(st.session_state.nodes, st.session_state.edges)


    # Close the database connection when done
        st.session_state.db.close()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("AIPatient Interface")
    st.text("Welcome to AIPatient: A Virtual Patient for Medical Education")
    main()