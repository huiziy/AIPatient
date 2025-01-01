from QA_generation.QA_generation_function.prompts import *
from ablation_study.ablation_study_function.prompts import *
import pandas as pd
from config import config
from itertools import product
import itertools
import statsmodels.api as sm
from statsmodels.formula.api import ols

class RobustnessEval:
    def __init__(self, db, llm, question_set):
        self.db = db
        self.llm = llm
        self.question_set = question_set
        self.sampled_question_set = self.sample_data(question_set, frac = 50/244)
        self.data_path = config["data_path"]
        self.final_data = {}
        self.schema = self.db.generate_schema()
        # Specify personality traits
        self.big_five_traits = {
            "Openness": ["Practical, conventional, prefers routine", "Curious, wide range of interests, independent"],
            "Conscientiousness": ["Impulsive, careless, disorganized", "Hardworking, dependable, organized"],
            "Extraversion": ["Quiet, reserved, withdrawn", "Outgoing, warm, seeks adventure"],
            "Agreeableness": ["Critical, uncooperative, suspicious", "Helpful, trusting, empathetic"],
            "Neuroticism": ["Calm, even-tempered, secure", "Anxious, unhappy, prone to negative emotions"]
            }
        # Generate all combinations
        all_combinations = list(itertools.product(*self.big_five_traits.values()))

        # Convert to list of dictionaries to represent each personality type
        self.list_of_personalities = []
        for combination in all_combinations:
            personality = dict(zip(self.big_five_traits.keys(), combination))
            self.list_of_personalities.append(personality)
            
    ## Data Sampler
    def sample_data(self, data, frac):
        """
        Samples a subset of data while preserving the proportion of `Question Category`.
        """
        return data.groupby('Question Category').apply(
            lambda x: x.sample(frac=frac, random_state=42)
        ).reset_index(drop=True)

        
    def run_model_for_flags(self, question_set, counter, personality):
        # Create empty results DataFrame to store ablation results
        results = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Question', 'CorrectAnswer', 'Query', 'Query_Result', 'Rewrite_Original',
                                        'Personality', 'Rewrite', 'GPT_Reasoning', 'Final_Answer'])

        for idx, row in question_set.iterrows():
            print(idx)
            subject_id = row['SUBJECT_ID']
            hadm_id = row['HADM_ID']
            question = row['Question']

            # Assuming cypher_query is defined elsewhere or is dynamic
            cypher_query = row["Cypher Query"]
            query_result = self.db.execute_cypher_query(cypher_query)

            # Original rewrite to natural language
            rewrite_result_prompt = rewrite_response_prompt(question, query_result)
            rewrite_result = self.llm.run_gpt(rewrite_result_prompt)

            personality_description = ', '.join(personality.values())
            print(personality_description)

            rewrite_result_prompt = rewrite_response_prompt(question, query_result, personality_trait=personality_description)
            rewrite = self.llm.run_gpt(rewrite_result_prompt)
            print(rewrite)

            # Check if are equivalent
            checker_prompt_text = checker_prompt_construction(result=rewrite_result, answer=rewrite)
            checker_result = self.llm.run_gpt(checker_prompt_text)
            GPT_reasoning = checker_result
            Final_Answer = extract_string_between_brackets(checker_result)

            # Append results to the dataframe
            new_row = {
                'idx': idx,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'Question': question,
                'CorrectAnswer': row['Correct Answer'],
                'Query': cypher_query,
                'Query_Result': query_result,
                'Rewrite_Original': rewrite_result,
                'Personality': personality,
                'Rewrite': rewrite,
                'GPT_Reasoning': GPT_reasoning,
                'Final_Answer': Final_Answer
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            print("saving now")
            results.to_csv(f"{self.data_path}/Robustness_Results/Personality_{counter}.csv", index=False)
            print("saved")

        return results
    
    def orchestrator(self):
        # Sample Subset of Question
        counter = 0
        for personality in self.list_of_personalities:
            counter += 1
            file_name = f"Bias_Evaluation_Paraphrase_{counter}.csv"
            print(file_name)
            results = self.run_model_for_flags(self.sampled_question_set, counter, personality)
            print(f"Results for personality {personality} stored.")
            
    def evaluator(self):
        merged_data = pd.DataFrame()
        file_accuracies = {}

        for counter in range(1, 33):  # from 1 to 32 inclusive
            # Construct the file path
            file_path = f"{self.data_path}/Robustness_Results/Annotated/Personality_{counter}.csv"
            # Read the CSV file
            data = pd.read_csv(file_path)
            # Add a column to track the Personality file source
            data['Personality'] = f"Personality_{counter}"
            # Merge with the existing merged_data DataFrame
            if merged_data.empty:
                merged_data = data
            else:
                merged_data = pd.concat([merged_data, data], ignore_index=True)

            # Calculate accuracy for the file
            accuracy = data['Final_Answer'].mean()  # Proportion of True answers
            file_accuracies[f"Personality_{counter}"] = accuracy  # Store accuracy in dictionary

        # Ensure 'Final_Answer' is boolean and convert it to integer for ANOVA
        merged_data['Final_Answer'] = merged_data['Final_Answer'].astype(bool).astype(int)

        # Perform ANOVA
        formula = 'Final_Answer ~ C(Personality)'
        model = ols(formula, data=merged_data).fit()
        anova_results_overall = sm.stats.anova_lm(model, typ=2)

        return file_accuracies, anova_results_overall
        
