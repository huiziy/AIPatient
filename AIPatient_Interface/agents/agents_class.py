import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time
from agents.agents_functions.prompts import *
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from streamlit_chat import message
from py2neo import Graph, Node as Py2neoNode, Relationship as Py2neoRelationship
from streamlit_agraph import agraph, Node, Edge, Config
from Neo4jDatabase.Neo4jDatabase_class import Neo4jDatabase
import logging
import os
import re


class Agents:
    def __init__(self,db, llm, model_type):
        self.db = db
        self.llm = llm
        self.model_type = model_type
        self.schema = self.db.generate_schema()

    ## Design Interface
    def interactive_session(self, doctor_query, conversation_history, patient_admission, personality_profile, max_token = 4096):
        if doctor_query.lower() == 'exit':
            logging.info("Session terminated by the user.")
            return "Session terminated by the user."
        
        #########################################################################################################################
        
        ## Step 2.1: Extract relevant nodes and edges
        logging.info("Extract relevant nodes and edges based on query.")
        nodes_edges_query_cypher_prompt = relationship_extraction_prompt(conversation_history, doctor_query, patient_admission, self.schema)
        nodes_edges_results = self.llm.run_model(nodes_edges_query_cypher_prompt, self.model_type)
        logging.info(f"Nodes and edges extracted: {nodes_edges_results}")

        ## Step 1: Construct Abstraction Query Prompt
        logging.info("Step 1: Constructing Abstraction Cypher query prompt based on the doctor's query.")
        abstraction_query_prompt = abstraction_generation_prompt(conversation_history, doctor_query)
        abstraction_query_nl = self.llm.run_model(abstraction_query_prompt, self.model_type)
        logging.info(f"Abstraction query in natural language generated: {abstraction_query_nl}")

        ## Step 3: Generate Abstraction Cypher Query
        logging.info("Constructing Cypher query prompt based on the abstraction query.")
        abstraction_query_cypher_prompt = cypher_query_construction_prompt(conversation_history, abstraction_query_nl, patient_admission, nodes_edges_results, self.schema)
        abstraction_query_cypher = self.llm.run_model(abstraction_query_cypher_prompt, self.model_type)
        logging.info(f"Abstraction cypher generated: {abstraction_query_cypher}")
        
        ## Step 3.5: Clean Cypher Query
        abstraction_query_cypher = clean_cypher_query(abstraction_query_cypher)

        ## Step 4: Execute the generated Cypher query
        logging.info("Step 4: Executing the generated Cypher query.")
        abstraction_result = self.db.execute_cypher_query(abstraction_query_cypher)
        if abstraction_result:
            ## Rewrite to natural language
            abstraction_result_rewrite_prompt = query_result_rewrite(abstraction_query_nl, abstraction_query_cypher, abstraction_result)
            abstract_result = self.llm.run_model(abstraction_result_rewrite_prompt, self.model_type)
        
        logging.info(f"Abstraction Query result: {abstraction_result}")

        #########################################################################################################################

        ## Step One: Original doctor's query
        logging.info(f"Step Zero: The doctors has asked about: {doctor_query}")
        logging.info("Step One: Constructing Cypher query prompt based on the doctor's query.")
        cypher_query_prompt = cypher_query_construction_prompt(conversation_history, doctor_query, patient_admission, nodes_edges_results, self.schema, abstraction_context=abstraction_result)

        ## Step 2.2: Construct Cypher Query
        cypher_query = self.llm.run_model(cypher_query_prompt, self.model_type)
        logging.info(f"Cypher query generated: {cypher_query}")
        
        ## Step 2.3: Clean Cypher Query
        cypher_query = clean_cypher_query(cypher_query)

        ## Step Three: Execute the generated Cypher query
        logging.info("Step Three: Executing the generated Cypher query.")
        query_result = self.db.execute_cypher_query(cypher_query)
        print(query_result)
        if query_result:
            print("Rewriting Retrieved Results to Natural Language")
            ## Rewrite to natural language
            query_result_rewrite_prompt = query_result_rewrite(doctor_query, cypher_query, query_result)
            query_result = self.llm.run_model(query_result_rewrite_prompt, self.model_type)
            print(f"The patient's response is {query_result}")
        logging.info(f"Query result: {query_result}")

        ## Step Four: Evaluate if the query properly answered the question
        for attempt in range(2):
            logging.info(f"Attempt {attempt + 1}: Evaluating the query result.")
            conversation_history = conversation_history.strip() if conversation_history else "No conversation history."
            doctor_query = doctor_query.strip() if doctor_query else "No query provided."
            query_result = query_result.strip() if isinstance(query_result, str) else str(query_result)

            checker_prompt = checker_construction_prompt(doctor_query, query_result, conversation_history)
            checked_result = self.llm.run_model(checker_prompt, self.model_type)
            logging.info(f"Checked result: {checked_result}")

            ## Process the checked result to separate the decision, reasoning, and rewritten query
            try:
                decision, rewritten_query, reasoning = process_checker_response(checked_result)
                print(f"Decision is {decision}")
                print(f"Reasoning is {reasoning}")
            except ValueError as e:
                logging.error(f"Error processing checker response: {e}")
                logging.info("Stopping evaluation due to unexpected response format.")
                break

            ## If the answer is deemed appropriate, stop the loop
            if decision == 'Y':
                logging.info(f"Checked result is appropriate. Reasoning: {reasoning}")
                break

            ## If the answer is deemed inappropriate, restructure the question and try again
            logging.info(f"Checked result is inappropriate. Reasoning: {reasoning}")
            logging.info(f"Rewriting the query: {rewritten_query}")
            cypher_query_prompt = cypher_query_construction_prompt(conversation_history, rewritten_query, patient_admission, nodes_edges_results, self.schema)
            cypher_query = self.llm.run_model(cypher_query_prompt, self.model_type)
            query_result = self.db.execute_cypher_query(cypher_query)
            query_result_rewrite_prompt = query_result_rewrite(doctor_query, cypher_query, query_result)
            query_result = self.llm.run_model(query_result_rewrite_prompt, self.model_type)
            logging.info(f"New query result: {query_result}")
            # if not query_result or len(query_result) == 0:
            #     query_result = ["I don't know"]
            #     logging.info("No appropriate answer after restructuring. Setting query result to 'I don't know'.")
            #     break

        ## If after three rounds, still no appropriate answer, return "I don't know."
        if decision != 'Y':
            query_result = ["I don't know"]
            logging.info("After two rounds, still no appropriate answer. Returning 'I don't know'.")

        ## Step Five: Given Query Results, generate the patient response
        logging.info("Step Five: Generating the patient response.")
        if query_result == ["I don't know"]:
            patient_response = "I don't know"
        else:
            rewrite_prompt = rewrite_response_prompt(conversation_history, doctor_query, query_result, patient_admission, personality_profile)
            patient_response = self.llm.run_model(rewrite_prompt, self.model_type)
            logging.info(f"Patient response generated: {patient_response}")

        ## Step Six: Update the conversation history
        logging.info("Step Six: Updating the conversation history.")
        summarization_prompt = summarize_text_prompt(conversation_history, doctor_query, patient_response)
        summarization = self.llm.run_model(summarization_prompt, self.model_type)
        logging.info(f"Conversation history updated: {summarization}")

        ## Update the conversation history based on the most recent interaction
        conversation_history = summarization
        logging.info(f"Conversation history: {conversation_history}")

        # Close the database connection
        self.db.close()
        logging.info("Database connection closed.")
        
        return patient_response, conversation_history
    
    def display_conversation(self, chat_placeholder):
        """Display the conversation in the chat placeholder."""
        with chat_placeholder.container():
            for idx, (doc_query, pat_response) in enumerate(st.session_state.conversation):
                timestamp = time.time()
                message(doc_query, is_user=True, key=f"doc_{idx}_{timestamp}", logo='https://raw.githubusercontent.com/huiziy/AIPatient_Image/master/doctor.png')
                message(pat_response, key=f"pat_{idx}_{timestamp}", logo='https://raw.githubusercontent.com/huiziy/AIPatient_Image/master/patient.png')

