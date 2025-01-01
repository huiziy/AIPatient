import re 
# Edge and node extraction
def relationship_extraction_prompt(conversation_history, text, patient_admission, schema):
    subject_id = patient_admission['SubjectID']
    hadm_id = patient_admission['AdmissionID']
    prompt = f"""
    
    Based on the doctor's query, first determine what the doctor is asking for. Then extract the appropriate relationship and nodes from the knowledge graph. \n
    For admissions related queries, the query should focus on "HAS_ADMISSION" relationship and "Admission" node. \n
    For patient information related queries, the query should focus on the "Patient" node. \n
    If the doctor asked about a symptom (e.g. cough, fever, etc.), the query should check if the "symptom" node and the "HAS_SYMPTOM" or "HAS_NOSYMOTOM" relationship; \n
    If the doctor asked about the duration, frequency, and intensity of a symptom, the query should first check if the symptom exist. If it exist, then check the "duration", "frequency" and "intensity" node respectively, and "HAS_DURATION", "HAS_FREQUENCY", "HAS_INTENSITY" relationship respectively. \n
    If the doctor asked about medical history, the query should check "History" node and the HAS_MEDICAL_HISTORY relationship. \n
    If the doctor asked about vitals (temperature, blood pressure etc), the query should check the "Vital" node and "HAS_VITAL" relationship. \n
    If the doctor asked about social history (smoking, alcohol consumption etc), the query should check the "SocialHistory" node and "HAS_SOCIAL_HISTORY" relationship. \n
    If the doctor aksed about family history, the query should first check the "HAS_FAMILY_MEMBER" relationship and "FamilyMember" node. Then, the query should check the "HAS_MEDICAL_HISTORY" relationship and "FamilyMedicalHistory" node associated with the "FamilyMember" node. \n 
    Output_format: Enclose your output in the following format. Do not give any explanations or reasoning, just provide the answer. For example:
    {{'Nodes': ['symptom', 'duration'], 'Relationships': ['HAS_SYMPTOM', 'HAS_DURATION']}}
    
    The natural language query is:
    {text}
    
    The previous conversation history is:
    {conversation_history}
    
    
    The Knowledge Graph Schema is:
    {schema}
    
    """
    
    return prompt
    

def cypher_query_construction_prompt(conversation_history, text, patient_admission, nodes_edges, schema, abstraction_context = None):
    subject_id = patient_admission['SubjectID']
    hadm_id = patient_admission['AdmissionID']
    prompt = f"""
    Write a cypher query to extract the requested information from the natural language query. The SUBJECT_ID is {subject_id}, and the HADM_ID is {hadm_id}.
    The nodes and edges the query should focus on are {nodes_edges} \n
    Note that if the doctor's query is vague, it should be referring to the current context.\n
    The Cypher query should be case insensitive and check if the keyword is contained in any fields (no need for exact match). \n
    The Cypher query should handle fuzzy matching for keywords such as 'temperature', 'blood pressure', 'heart rate', etc., in the LABEL attribute of Vital nodes.\n
    The Cypher query should also handel matching smoke, smoking, tobacco if asked about smoking and social history; similarly for drinking, or alcohol. \n
    Only return the query as it should be executable directly, and no other text. Don't include any new line characters, or back ticks, or the word 'cypher', or square brackets, or quotes.\n
    
    The previous conversation history is:
    {conversation_history}
    
    The natural language query is:
    {text}
    
    The Knowledge Graph Schema is:
    {schema}"""

    if abstraction_context is not None:
        prompt += f"""
        The step back context is:
        {abstraction_context} 
    """

    prompt += """
    Here are a few examples of Cypher queries, you should replay SUBJECT_ID and HADM_ID based on input:\n
    
    Example 1: To list all the symptoms the patient has, the cypher query should be: 
    MATCH (p:Patient {{SUBJECT_ID: '23709'}})
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(:Admission)-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN collect(s.name) AS symptoms
    
    Example 2: To check if the patient has seizures as a symptom, the cypher query should be: 
    MATCH (p:Patient {{SUBJECT_ID: '23709'}})
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission {{HADM_ID: '182203'}})-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name =~ '(?i).*seizure.*'
    WITH p, a, s
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a)-[:HAS_NOSYMPTOM]->(ns:Symptom)
    WHERE ns.name =~ '(?i).*seizure.*'
    RETURN 
    CASE 
        WHEN s IS NOT NULL THEN 'HAS seizure'
        WHEN ns IS NOT NULL THEN 'DOES NOT HAVE seizures'
        ELSE 'DONT KNOW'
    END AS status

    Example 3: To check how long has the patient had fevers as a symptom, the cypher query should be:
    MATCH (p:Patient {{SUBJECT_ID: '23709'}})
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission {{HADM_ID: '182203'}})-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name =~ '(?i).*fever.*'
    WITH p, a, s
    OPTIONAL MATCH (s)-[:HAS_DURATION]->(d:Duration)
    RETURN 
    CASE 
        WHEN s IS NULL THEN 'DOES NOT HAVE fevers'
        WHEN d IS NULL THEN 'DONT KNOW'
        ELSE d.name
    END AS fever_duration

    Example 4: To check for family history, the cupher query should be: 
    MATCH (p:Patient {{SUBJECT_ID: '23709'}})-[:HAS_FAMILY_MEMBER]->(fm:FamilyMember)
    OPTIONAL MATCH (fm)-[:HAS_MEDICAL_HISTORY]->(fmh:FamilyMedicalHistory)
    RETURN fm.name AS family_member, fmh.name AS medical_history
    
    Formatting Instructions:
    - Queries are case-insensitive and should check if the keyword is contained in any relevant fields (no exact match needed).
    - Handle fuzzy matching for terms like 'temperature', 'blood pressure', and 'heart rate' in the Vital nodes' LABEL attribute.
    - Also handle fuzzy matching for 'smoke', 'smoking', or 'tobacco' in social history; similarly for 'drinking' or 'alcohol' regarding alcohol consumption.
    - Use single quotes in the Cypher query for SUBJECT_ID and HADM_ID.


    This is important: You MUST format the query so it can be executed directly, without any reasoning explanation, new line characters, backticks, the word 'cypher', square brackets, or additional quotes. When checking for keywords, the query MUST be case insensitive. 
    """
    return prompt

## Helper function: format cypher query
def clean_cypher_query(query):
    # Remove surrounding quotes
    query = query.strip('"')
    # Remove surrounding brackets
    query = query.strip('[]')
    # Remove newline characters
    query = query.replace('\\n', ' ')
    # Remove any leading or trailing whitespace characters
    query = query.strip()
    # Normalize whitespace within the query
    query = re.sub(r'\s+', ' ', query)
    return query

## Abstraction with a Few Shot Examples
def abstraction_generation_prompt(conversation_history, text):
    prompt = f"""
    You are an AI and Medical EHR expert. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to use for cypher query generation. \n 
    If the question is vague, consider the conversation history and the current context. Do not give any explanations or reasoning, just provide the answer. 
    Here are a few examples: \n
    input: Do you have fevers as a symptom? \n
    output: What symptoms does the patient has? \n
    input: Is your current temperature above 97 degrees? \n
    output: What is the patient's temperature? \n
    
    The current conversation history is:
    {conversation_history}
    The original query is:
    {text}
    """
    return prompt

## Rewrite Query Result Function
## This function combines the query results, and relationship to convert it to natural language
def query_result_rewrite(doctor_query, cypher_query, query_result):
    prompt = f"""
    You are a doctor's assistant. Based on the cypher_query, please structure the retrieved query results into natural language. Include all subject, relationship and object. 
    For example: \n
    doctor query: what symptoms do you have?
    cypher query: MATCH (p:Patient)-[:HAS_ADMISSION]->(a:Admission {{HADM_ID: 182203}})
    MATCH (a)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE p.SUBJECT_ID = '23709'
    RETURN s.name AS Symptom 

    retrieved result: ['black and bloody stools', 'lightheadedness', 'shortness of breath']

    output: The patient has symptoms of black and bloody stools, lightheadedness, shortness of breath. 

    The doctor's original query is:
    {doctor_query}
    The cypher query is:
    {cypher_query}
    The retrieved results are:
    {query_result}
    """

    return prompt


## Summarization Function
def summarize_text_prompt(conversation_history, doctor_query, patient_response):
    prompt = f"""
    You are the doctor's assistent responsible for summarizing the conversation between the doctor and the patient.
    Be very brief, include the all the conversation history, doctor and patient's query and response. The last sentence should be about the current context (e.g. vital, symptom, or history).
    Write in full sentences and do not fabricate symptoms or history.
    The previous conversation is as follows:
    {conversation_history}
    The doctor has asked about the following query:
    {doctor_query}
    The patient's response to the doctor's query:
    {patient_response}
    """
    return prompt

## Rewrite Function
def rewrite_response_prompt(conversation_history, doctor_query, query_result, patient_admission, personality):
    subject_id = patient_admission['SubjectID']
    hadm_id = patient_admission['AdmissionID']
    prompt = f"""
    You are a virtual patient in an office visit. Your personality is {personality}.
    Your conversation history with the doctor is as follows:
    {conversation_history}
    The doctor has asked about the following query, focusing on the current context (e.g. vital, symptom, or history):
    {doctor_query}
    The patient's response to the doctor' query is:
    {query_result}
    Based on all above information, please write your response to the doctor following your personality traits. Note that if the doctor's query is vague, it should be referring to the current context.
    If the patient's response is empty, return 'I don't know.' DO NOT fabricate any symptom or medical history. DO NOT add non-existent details to the response. DO NOT inclue any quotes, write in first person perspective. 
    """
    return prompt

def checker_construction_prompt(doctor_query, query_result, conversation_history):
    prompt = f"""
    You are a doctor's assistant. You are recording and evaluating the patient's responses to the doctor's query.
    The conversation history between the doctor and patient is as follows:
    {conversation_history}
    The doctor's query is:
    {doctor_query}
    The patient's response to the doctor's query is:
    {query_result}
    Based on the above conversation:
    - If the patient's response is appropriate, return 'Y' followed by reasoning explaining why it is appropriate.
    - If the patient's response is not appropriate, return 'N', the rewritten query, and reasoning explaining why the original response was inadequate.
    - DO NOT write anything else, Output EXACTLY as in the format below. 
    Use the following format:
    - 'Y: <reasoning>'
    - 'N: <rewritten query> <REASONING_TAG> <reasoning>'
    """
    return prompt

def process_checker_response(response):
    response = response.strip()
    if response.startswith("Y:"):
        decision = "Y"
        reasoning = response[2:].strip()  # Extract reasoning after 'Y:'
        rewritten_query = None
    elif response.startswith("N:"):
        decision = "N"
        # Split using the <REASONING_TAG> tag
        try:
            parts = response[2:].split('<REASONING_TAG>')
            rewritten_query = parts[0].strip()
            reasoning = parts[1].strip() if len(parts) > 1 else "No reasoning provided."
        except IndexError:
            raise ValueError("Unexpected response format: unable to separate query and reasoning.")
    else:
        raise ValueError("Unexpected response format from checker.")
    
    return decision, rewritten_query, reasoning