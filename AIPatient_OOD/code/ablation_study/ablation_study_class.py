import pandas as pd
from config import config
from openai import OpenAI
from itertools import product
from ablation_study.ablation_study_function.prompts import *

class AblationEval:
    def __init__(self, db, llm, question_set):
        self.db = db
        self.llm = llm
        self.question_set = question_set
        self.data_path = config["data_path"]
        self.final_data = {}
        self.schema = self.db.generate_schema()
        
    
    def run_model_for_flags(self, fewshot, retrieval_agent, abstraction_agent, filename):
        results_df = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Query', 'FailureFlag', 'Question', 'CorrectAnswer', 'RetrievedResult', "GPTReasoning", "Final_Answer"])
        for idx, row in self.question_set.iterrows():
            try:
                print(idx)
                subject_id = row['SUBJECT_ID']
                hadm_id = row['HADM_ID']
                question = row['Question']
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

                    abstraction_query_cypher_prompt = cypher_query_construction_prompt(abstraction_query_nl, self.schema, patient_admission, nodes_edges=nodes_edges_results, retrieval_agent=retrieval_agent)
                    abstraction_query_cypher = self.llm.run_gpt(abstraction_query_cypher_prompt)

                    abstraction_query_cypher = clean_cypher_query(abstraction_query_cypher)
                    abstraction_result = self.db.execute_cypher_query(abstraction_query_cypher)

                    if abstraction_result:
                        abstraction_result_rewrite_prompt = rewrite_response_prompt(abstraction_query_nl, abstraction_result)
                        abstraction_result = self.llm.run_gpt(abstraction_result_rewrite_prompt)

                # Cypher Query Prompt
                prompt = cypher_query_construction_prompt(question, self.schema, patient_admission, nodes_edges=nodes_edges_results, abstraction_context=abstraction_result, fewshot=fewshot, retrieval_agent=retrieval_agent)
                cypher_query = self.llm.run_gpt(prompt)

                cypher_query = clean_cypher_query(cypher_query)
                query_result = self.db.execute_cypher_query(cypher_query)

                # Rewrite to natural language
                rewrite_result_prompt = rewrite_response_prompt(question, query_result)
                rewrite_result = self.llm.run_gpt(rewrite_result_prompt)

                rewrite_answer_prompt = rewrite_response_prompt(question, row["Correct Answer"])
                rewrite_answer = self.llm.run_gpt(rewrite_answer_prompt)

                checker_prompt_text = checker_prompt_construction(answer=rewrite_answer, result=rewrite_result)
                checker_result = self.llm.run_gpt(checker_prompt_text)
                GPT_reasoning = checker_result
                Final_Answer = extract_string_between_brackets(checker_result)
                print(Final_Answer)
                failure_flag = 0

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                cypher_query = None
                query_result = None
                rewrite_result = None
                rewrite_answer = None
                GPT_reasoning = None
                Final_Answer = None
                failure_flag = 1

            # Append results to the dataframe
            new_row = {
                'idx': idx,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'Query': cypher_query,
                'FailureFlag': failure_flag,
                'Question': question,
                'CorrectAnswer': rewrite_answer,
                'RetrievedResult': rewrite_result,
                'GPTReasoning': GPT_reasoning,
                'Final_Answer': Final_Answer
            }
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            results_df.to_csv(filename, index=False)
            iteration_name = f"fewshot_{fewshot}_retrieval_{retrieval_agent}_abstraction_{abstraction_agent}"
        
        self.final_data[iteration_name] = results_df
        return results_df
    
    
    def orchestrator(self):
        #combinations = list(product([True, False], repeat=3))
        combinations = list(product([True, False], repeat=3))
        for fewshot, retrieval_agent, abstraction_agent in combinations:
            print(f"Running with fewshot={fewshot}, retrieval_agent={retrieval_agent}, abstraction_agent={abstraction_agent}")
            filename = f"{self.data_path}/Ablation_Results/results_fewshot_{fewshot}_retrieval_{retrieval_agent}_abstraction_{abstraction_agent}.csv"
            results = self.run_model_for_flags(fewshot, retrieval_agent, abstraction_agent, filename)
            # Calculate accuracy rate
            accuracy_rate = (results['Final_Answer'] == 'True').sum() / len(results)
            print(f"Accuracy Rate for fewshot={fewshot}, retrieval_agent={retrieval_agent}, abstraction_agent={abstraction_agent}: {accuracy_rate}")
            
            
    def evaluator(self):
        all_results = []
        category_order = [
        'Admission', 'Patient', 'Symptom', 'Medical History', 'Allergy', 'Family and Social History'
    ]

        category_mapping = {
        'admission': 'Admission',
        'allergy': 'Allergy',
        'familyhistory': 'Family and Social History',
        'medicalhistory': 'Medical History',
        'patient': 'Patient',
        'socialhistory': 'Family and Social History',
        'symptom': 'Symptom'
    }
        ## Merge on original dataset for the question category 
        combinations = list(product([True, False], repeat=3))
        for fewshot, retrieval_agent, abstraction_agent in combinations:
            filename = f"{self.data_path}/Ablation_Results/results_fewshot_{fewshot}_retrieval_{retrieval_agent}_abstraction_{abstraction_agent}.csv"
            data = pd.read_csv(filename)
            data = data.merge(self.question_set, on=["Question", "SUBJECT_ID", "HADM_ID"])
            data['Question Category'] = data['Question Category'].replace(category_mapping)
            data['Question Category'] = pd.Categorical(data['Question Category'], categories=category_order, ordered=True)
            data = data.sort_values('Question Category').reset_index(drop=True)

            total_rows = len(data)
            true_count = data['Final_Answer'].sum()
            overall_proportion = true_count / total_rows if total_rows > 0 else 0

            proportion_true_by_category = (
                data.groupby('Question Category')['Final_Answer']
                .apply(lambda x: x.sum() / len(x))
                .reset_index(name='Proportion Correct')
            )

            pivoted = proportion_true_by_category.pivot_table(index=None, columns='Question Category', values='Proportion Correct').reset_index(drop=True)
            pivoted = pivoted.applymap(lambda x: f"{(x * 100):.2f}%" if pd.notnull(x) else x)
            pivoted['Overall'] = f"{(overall_proportion * 100):.2f}%"

            # Add columns for fewshot, retrieval_agent, and abstraction_agent
            pivoted['Fewshot'] = fewshot
            pivoted['Retrieval Agent'] = retrieval_agent
            pivoted['Abstraction Agent'] = abstraction_agent

            # Append to the list of results
            all_results.append(pivoted)

        # Concatenate all results into a single dataframe
        final_df = pd.concat(all_results, ignore_index=True)
        self.final_data["ablation_result"] = final_df
        return (final_df)

            
        

        
