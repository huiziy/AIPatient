from QA_generation.QA_generation_function.prompts import *
from ablation_study.ablation_study_function.prompts import *
import pandas as pd
from config import config
from itertools import product
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

class StabilityEval:
    def __init__(self, db, llm, question_set):
        self.db = db
        self.llm = llm
        self.question_set = question_set
        self.data_path = config["data_path"]
        self.final_data = {}
        self.schema = self.db.generate_schema()
        
        
    def run_model_for_flags(self,question_set, fewshot, retrieval_agent, abstraction_agent, base_filename):
        # create empty results df to store ablation results
        # Create three empty results dataframes to store results for each paraphrased question
        results_1 = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Query', 'FailureFlag', 'Question', 'Paraphrased_Question', 'CorrectAnswer', 'RetrievedResult', "GPTReasoning", "Final_Answer"])
        results_2 = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Query', 'FailureFlag', 'Question', 'Paraphrased_Question', 'CorrectAnswer', 'RetrievedResult', "GPTReasoning", "Final_Answer"])
        results_3 = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Query', 'FailureFlag', 'Question', 'Paraphrased_Question', 'CorrectAnswer', 'RetrievedResult', "GPTReasoning", "Final_Answer"])

        results_dfs = [results_1, results_2, results_3]

        for idx, row in question_set.iterrows():
            try:
                print(f"Processing row {idx}")
                subject_id = row['SUBJECT_ID']
                hadm_id = row['HADM_ID']
                question_original = row['Question']
                patient_admission = {
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': hadm_id
                }
                question_category = row["Question Category"]
                ## Collect node info
                node_info = combine_patient_information(self.db, subject_id, hadm_id)

                # Generate paraphrased questions
                paraphrase_question_prompt = paraphrase_question_agent(question_original, self.schema, node_info, patient_admission, question_category)
                paraphrase_response = self.llm.run_gpt(paraphrase_question_prompt)
                # Parse the paraphrased questions from the response
                paraphrased_questions = parse_paraphrased_questions(paraphrase_response)
                print(f"Paraphrased questions for row {idx}: {paraphrased_questions}")

                for i, question in enumerate(paraphrased_questions):
                    print(f"Processing paraphrase {i+1}: {question}")

                    nodes_edges_results = None
                    abstraction_result = None

                    patient_admission = {
                        'SubjectID': subject_id,
                        'AdmissionID': hadm_id
                    }

                    # Retrieve nodes and edges
                    if retrieval_agent:
                        nodes_edges_query_cypher_prompt = relationship_extraction_prompt(question, self.schema)
                        nodes_edges_results = self.llm.run_gpt(nodes_edges_query_cypher_prompt)

                    # Abstraction Generation
                    if abstraction_agent:
                        abstraction_query_prompt = abstraction_generation_prompt(question)
                        abstraction_query_nl = self.llm.run_gpt(abstraction_query_prompt)

                        abstraction_query_cypher_prompt = cypher_query_construction_prompt(
                            abstraction_query_nl, self.schema, patient_admission, nodes_edges=nodes_edges_results, retrieval_agent=retrieval_agent)
                        abstraction_query_cypher = self.llm.run_gpt(abstraction_query_cypher_prompt)

                        abstraction_query_cypher = clean_cypher_query(abstraction_query_cypher)
                        abstraction_result = self.db.execute_cypher_query(abstraction_query_cypher)

                        if abstraction_result:
                            abstraction_result_rewrite_prompt = rewrite_response_prompt(abstraction_query_nl, abstraction_result)
                            abstraction_result = self.llm.run_gpt(abstraction_result_rewrite_prompt)

                    # Cypher Query Prompt
                    prompt = cypher_query_construction_prompt(
                        question, self.schema, patient_admission, nodes_edges=nodes_edges_results, abstraction_context=abstraction_result,
                        fewshot=fewshot, retrieval_agent=retrieval_agent)
                    cypher_query = self.llm.run_gpt(prompt)

                    # Clean result if it is Claude generated
                    cypher_query = clean_cypher_query(cypher_query)
                    query_result = self.db.execute_cypher_query(cypher_query)

                    # Rewrite to natural language
                    rewrite_result_prompt = rewrite_response_prompt(question, query_result)
                    rewrite_result = self.llm.run_gpt(rewrite_result_prompt)

                    rewrite_answer_prompt = rewrite_response_prompt(question, row["Correct Answer"])
                    rewrite_answer = self.llm.run_gpt(rewrite_answer_prompt)

                    # Check if answers are equivalent
                    checker_prompt_text = checker_prompt_construction(answer=rewrite_answer, result=rewrite_result)
                    checker_result = self.llm.run_gpt(checker_prompt_text)
                    GPT_reasoning = checker_result
                    Final_Answer = extract_string_between_brackets(checker_result)

                    failure_flag = 0  # Set the failure flag to 0 since no exception occurred

                    # Append results to the dataframe
                    new_row = {
                        'idx': idx,
                        'SUBJECT_ID': subject_id,
                        'HADM_ID': hadm_id,
                        'Query': cypher_query,
                        'FailureFlag': failure_flag,
                        'Question': question_original,
                        'Paraphrased_Question': question,
                        'CorrectAnswer': rewrite_answer,
                        'RetrievedResult': rewrite_result,
                        'GPTReasoning': GPT_reasoning,
                        'Final_Answer': Final_Answer
                    }
                    results_dfs[i] = pd.concat([results_dfs[i], pd.DataFrame([new_row])], ignore_index=True)
                    accuracy_rate = (results_dfs[i]['Final_Answer'] == 'True').sum() / len(results_dfs[i])
                    print(f"Accuracy Rate for Paraphrase {i+1}: {accuracy_rate:.2%}")

                    # Save the results for each paraphrase
                    filename = f"{base_filename}_paraphrase_{i+1}.csv"
                    results_dfs[i].to_csv(filename, index=False)

            except Exception as e:
                print(f"Error processing row {idx} paraphrase {i+1}: {e}")
                cypher_query = None
                query_result = None
                rewrite_result = None
                rewrite_answer = None
                GPT_reasoning = None
                Final_Answer = None
                failure_flag = 1  # Set the failure flag to 1 since an exception occurred

                # Append error results to the dataframe
                new_row = {
                    'idx': idx,
                    'SUBJECT_ID': subject_id,
                    'HADM_ID': hadm_id,
                    'Query': cypher_query,
                    'FailureFlag': failure_flag,
                    'Question': question_original,
                    'Paraphrased_Question': question,
                    'CorrectAnswer': rewrite_answer,
                    'RetrievedResult': rewrite_result,
                    'GPTReasoning': GPT_reasoning,
                    'Final_Answer': Final_Answer
                }
                results_dfs[i] = pd.concat([results_dfs[i], pd.DataFrame([new_row])], ignore_index=True)

        return results_dfs
    
    
    def orchestrator(self):
        #combinations = list(product([True, False], repeat=3))
        combinations = list(product([True, False], repeat=3))
        #choose the best combination
        #begin experiment   
        fewshot, retrieval_agent, abstraction_agent = combinations[0]
        ## Need to loop three times
        print(f"fewshot:{fewshot}, retrieval_agent:{retrieval_agent}, abstraction_agent:{abstraction_agent}")
        filename = f"{self.data_path}/Stability_Results/results_fewshot_{fewshot}_retrieval_{retrieval_agent}_abstraction_{abstraction_agent}"
        results_df = self.run_model_for_flags(self.question_set, fewshot, retrieval_agent, abstraction_agent, base_filename = filename)
        ## print the accuracy rate: the sum of value that is "True" in Final_Answer divided by the total number of rows
        for i, results in enumerate(results_df):
            accuracy_rate = results['Final_Answer'].str.contains(r'\bTrue\b', na=False).sum()
            print(f"Accuracy Rate for Paraphrase {i+1}: {accuracy_rate:.2%}")
            
    def perform_anova(self, data, group_by):
        formula = f'Final_Answer ~ C({group_by})'
        model = ols(formula, data=data).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA
        return anova_results
            
    def evaluator(self):
        # Load and label each dataset
        all_data = []
        # Dataset 1: Ablation Results
        ablation_df = pd.read_csv(f"{self.data_path}/Ablation_Results/results_fewshot_True_retrieval_True_abstraction_True.csv")
        ablation_df['Group'] = 'Ablation'
        all_data.append(ablation_df[['Final_Answer', 'Group']])

        # Dataset 2-4: Stability Results
        for i in range(1, 4):
            filename = f"{self.data_path}/Stability_Results/Final/results_fewshot_True_retrieval_True_abstraction_True_paraphrase_{i}.csv"
            stability_df = pd.read_csv(filename)
            stability_df['Group'] = f'Test_{i}'
            all_data.append(stability_df[['Final_Answer', 'Group']])

        # Concatenate all data into a single DataFrame
        all_data = pd.concat(all_data, ignore_index=True)

        # Ensure Final_Answer is binary (1 for correct, 0 for incorrect)
        all_data['Final_Answer'] = all_data['Final_Answer'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

        # Perform overall ANOVA
        overall_anova_results = self.perform_anova(all_data, 'Group')

        # Display results
        print("Overall ANOVA Results:")
        print(overall_anova_results)