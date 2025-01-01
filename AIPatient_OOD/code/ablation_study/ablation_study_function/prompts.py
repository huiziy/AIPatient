import re 

def cypher_query_construction_prompt(
    text, schema, patient_admission, nodes_edges=None, abstraction_context=None, 
    kg_schema=True, fewshot=False, retrieval_agent=False
):
    subject_id = patient_admission['SubjectID']
    hadm_id = patient_admission['AdmissionID']
    
    prompt = f"""
    Write a Cypher query to extract the requested information from the natural language query.
    The SUBJECT_ID is '{subject_id}', and the HADM_ID is '{hadm_id}'.
    """

    # Add retrieval agent specifics if required
    if retrieval_agent and nodes_edges:
        prompt += f"The nodes and edges the query should focus on are: {nodes_edges}\n\n"

    prompt += f"The natural language query is:\n{text}\n\n"

    # Provide the schema if available
    if kg_schema:
        prompt += f"The Knowledge Graph Schema is:\n{schema}\n\n"

    # Include abstraction context if available
    if abstraction_context:
        prompt += f"The context from the abstraction is:\n{abstraction_context}\n\n"

    # Add few-shot examples if specified
    if fewshot:
        prompt += """
    Here are a few examples of Cypher queries:

    Example 1: To check if the patient is male:
    MATCH (p:Patient {SUBJECT_ID: '23709'})
    RETURN p.GENDER =~ '(?i)male' AS isMale

    Example 2: To check if the patient has seizures as a symptom:
    MATCH (p:Patient {SUBJECT_ID: '23709'})
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission {HADM_ID: '182203'})-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name =~ '(?i).*seizure.*'
    WITH p, a, s
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a)-[:HAS_NOSYMPTOM]->(ns:Symptom)
    WHERE ns.name =~ '(?i).*seizure.*'
    RETURN CASE WHEN s IS NOT NULL THEN 'HAS seizure' WHEN ns IS NOT NULL THEN 'DOES NOT HAVE seizures' ELSE 'NO' END AS status

    Example 3: To check how long has the patient had fevers as a symptom:
    MATCH (p:Patient {SUBJECT_ID: '23709'})
    OPTIONAL MATCH (p)-[:HAS_ADMISSION]->(a:Admission {HADM_ID: '182203'})-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name =~ '(?i).*fever.*'
    WITH p, a, s
    OPTIONAL MATCH (s)-[:HAS_DURATION]->(d:Duration)
    RETURN CASE WHEN s IS NULL THEN 'DOES NOT HAVE fevers' WHEN d IS NULL THEN 'NO' ELSE d.name END AS fever_duration

    Example 4: To check for family history:
    MATCH (p:Patient {SUBJECT_ID: '23709'})-[:HAS_FAMILY_MEMBER]->(fm:FamilyMember)
    OPTIONAL MATCH (fm)-[:HAS_MEDICAL_HISTORY]->(fmh:FamilyMedicalHistory)
    RETURN fm.name AS family_member, fmh.name AS medical_history

    Example 5: To check all medical history:
    MATCH (p:Patient)-[r:HAS_MEDICAL_HISTORY]->(h:History)
    WHERE EXISTS((p)-[:HAS_ADMISSION]->(:Admission {HADM_ID: '182203'}))
    RETURN p, r, h
    """

    prompt += """
    Formatting Instructions:
    - Queries are case-insensitive and should check if the keyword is contained in any relevant fields (no exact match needed).
    - Handle fuzzy matching for terms like 'temperature', 'blood pressure', and 'heart rate' in the Vital nodes' LABEL attribute.
    - Also handle fuzzy matching for 'smoke', 'smoking', or 'tobacco' in social history; similarly for 'drinking' or 'alcohol' regarding alcohol consumption.
    - Use single quotes in the Cypher query for SUBJECT_ID and HADM_ID.


    This is important: You MUST format the query so it can be executed directly, without any reasoning explanation, new line characters, backticks, the word 'cypher', square brackets, or additional quotes. When checking for keywords, the query MUST be case insensitive. 
    """

    # Print the prompt for debugging
    # print(prompt)

    return prompt.strip()

#Do not return any reasoning or description. Don't include any new line characters, or back ticks, or square brackets, or quotes.\n
def checker_prompt_construction(answer, result):
    prompt = f"""
    Evaluate if the two provided answers have the same information; they do not have to match exactly.

    Your evaluation should be case insensitive, order insensitive, and handles fuzzy matching. For example, 'Y' and 'Yes' are equivalent; 'smoke' and 'smoking' are equivalent
    If they are equivalent, return 'True', otherwise return 'False'. Enclose your answer in <>. For example <Yes>.
    Provide reasoning.
    The provided answer is: {answer}
    The retrieved result is: {result}
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
def abstraction_generation_prompt(text):
    prompt = f"""
    You are an AI and Medical EHR expert. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to use for cypher query generation. \n
    If the question is vague, consider the conversation history and the current context. Do not give any explanations or reasoning, just provide the answer.
    Here are a few examples: \n
    input: Do you have fevers as a symptom? \n
    output: What symptoms does the patient has? \n
    input: Is your current temperature above 97 degrees? \n
    output: What is the patient's temperature? \n

    The original query is:
    {text}
    """
    return prompt

# Edge and node extraction
def relationship_extraction_prompt(text, schema):
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
    Output_format: Enclose your output in the following format. Do not give any explanations or reasoning, don't include quotes, or the word 'json', and just provide the answer. For example:
    {{'Nodes': ['symptom', 'duration'], 'Relationships': ['HAS_SYMPTOM', 'HAS_DURATION']}}

    The natural language query is:
    {text}

    The Knowledge Graph Schema is:
    {schema}
    """

    return prompt

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

# Rewrite Response
def rewrite_response_prompt(doctor_query, query_result, personality_trait=None):
    if personality_trait:
        prompt = f"""
        You are a virtual patient in an office visit. Your personality is {personality_trait}.
        The doctor has asked about the following query, focusing on the current context (e.g. vital, symptom, or history):
        {doctor_query}
        The information is:
        {query_result}
        If the query result is empty, consider the query result as 'No' and rewrite. DO NOT include any quotes.
        Based on all above information, please write your response as if you are role-playing (in first person) to the doctor following your personality traits.
        Please highlight your personality traits in your response. You can freely choose what information to disclose based on your personality traits. You don't need to provide all the information.
        For example, if the personality is uncooperative, you might be relunctant to provide information, or provide false information.
        """
    else:
        prompt = f"""
        You are a virtual patient in an office visit.
        The doctor has asked about the following query, focusing on the current context (e.g. vital, symptom, or history):
        {doctor_query}
        The query results are:
        {query_result}
        If the query result is empty, consider the query result as 'No' and rewrite. DO NOT fabricate any symptom or medical history. DO NOT add non-existent details to the response. DO NOT include any quotes.
        Based on all above information, please write your response as if you are role-playing (in first person) to the doctor.
        """
    return prompt

def extract_string_between_brackets(input_string):
    match = re.search(r'<(.*?)>', input_string)
    if match:
        return match.group(1)
    else:
        return None
    
    
def paraphrase_question_agent(input_question, schema, node_info, patient_info, question_category, num_paraphrases=3):

    prompt = f"""
    Please generate {num_paraphrases} different paraphrases of the following question while maintaining its original meaning. You should only ask about the nodes and edges in the schema.
    Each paraphrase should be unique and capture the essence of the original question in a different way. When paraphrasing, only use the keywords from the original query or the node information. Don't change the keywords.
    
    The patient's SUBJECT_ID is 
    {patient_info["SUBJECT_ID"]}
    
    The patient's HADM_ID is 
    {patient_info["HADM_ID"]}
    
    The question category is 
    {question_category}

    The schema is
    {schema}

    The node information is
    {node_info}

    The original question is:
    {input_question}

    Please return the paraphrased questions in the following format. Include the patient's SUBJECT_ID and HADM_ID, and question_category in the paraphrase question:
    <PARAPHRASE_1>
    [Insert first paraphrased question here]
    </PARAPHRASE_1>
    <PARAPHRASE_2>
    [Insert second paraphrased question here]
    </PARAPHRASE_2>
    <PARAPHRASE_3>
    [Insert third paraphrased question here]
    </PARAPHRASE_3>
    """
    return prompt


def parse_paraphrased_questions(response):
    pattern = r"<PARAPHRASE_1>(.*?)</PARAPHRASE_1>.*<PARAPHRASE_2>(.*?)</PARAPHRASE_2>.*<PARAPHRASE_3>(.*?)</PARAPHRASE_3>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
    else:
        return []