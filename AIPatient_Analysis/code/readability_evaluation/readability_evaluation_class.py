from QA_generation.QA_generation_function.prompts import *
from ablation_study.ablation_study_function.prompts import *
import pandas as pd
from config import config
import itertools
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import textstat

class ReadabilityEval:
    def __init__(self, db, llm, question_set):
        self.db = db
        self.llm = llm
        self.question_set = question_set
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
        
        
    def run_model_for_flags(self, filename):
        # Create empty results DataFrame to store ablation results
        results = pd.DataFrame(columns=['idx', 'SUBJECT_ID', 'HADM_ID', 'Query', 'Query_Result', 'Rewrite_Original',
                                        'Personality_1', 'Rewrite_1', 'GPT_Reasoning_1', 'Final_Answer_1', 'Personality_2',
                                        'Rewrite_2', 'GPT_Reasoning_2', 'Final_Answer_2','Personality_3', 'Rewrite_3', 'GPT_Reasoning_3', 'Final_Answer_3'])

        for idx, row in self.question_set.iterrows():
            try:
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

                # Generate three random personalities and corresponding rewrites
                personalities = []
                rewrites = []
                gpt_reasoning = []
                final_answer = []

                for _ in range(3):
                    random_personality = random.choice(self.list_of_personalities)
                    personalities.append(random_personality)
                    rewrite_result_prompt = rewrite_response_prompt(question, query_result, personality_trait=random_personality)
                    rewrite = self.llm.run_gpt(rewrite_result_prompt)
                    rewrites.append(rewrite)

                    # Check if are equivalent
                    checker_prompt_text = checker_prompt_construction(result=rewrite_result, answer=rewrite)
                    checker_result = self.llm.run_gpt(checker_prompt_text)
                    GPT_reasoning = checker_result
                    Final_Answer = extract_string_between_brackets(checker_result)
                    print(Final_Answer)
                    gpt_reasoning.append(GPT_reasoning)
                    final_answer.append(Final_Answer)

                failure_flag = 0  # Set the failure flag to 0 since no exception occurred

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                cypher_query = None
                query_result = None
                rewrite_result = None
                personalities = [None] * 3
                rewrites = [None] * 3
                gpt_reasoning = [None] * 3
                final_answer = [None] * 3
                failure_flag = 1  # Set the failure flag to 1 since an exception occurred

            # Append results to the dataframe
            new_row = {
                'idx': idx,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'Query': cypher_query,
                'Query_Result': query_result,
                'Rewrite_Original': rewrite_result,
                'Personality_1': personalities[0],
                'Rewrite_1': rewrites[0],
                'GPT_Reasoning_1': gpt_reasoning[0],
                'Final_Answer_1': final_answer[0],
                'Personality_2': personalities[1],
                'Rewrite_2': rewrites[1],
                'GPT_Reasoning_2': gpt_reasoning[1],
                'Final_Answer_2': final_answer[1],
                'Personality_3': personalities[2],
                'Rewrite_3': rewrites[2],
                'GPT_Reasoning_3': gpt_reasoning[2],
                'Final_Answer_3': final_answer[2]
            }
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            results.to_csv(filename, index=False)
            self.final_data["Readability_Evaluation_df"] = results

        return results
    
    def calculate_and_append_scores(self, df):
        # Define the columns to analyze
        rewrite_columns = ["Rewrite_Original", "Rewrite_1", "Rewrite_2", "Rewrite_3"]
        
        # Loop through each row and each rewrite column to calculate scores
        for index, row in df.iterrows():
            for rewrite in rewrite_columns:
                text = row[rewrite]
                if isinstance(text, str) and text.strip():  # Check if text is a non-empty string
                    # Calculate Flesch Reading Ease score
                    flesch_reading_ease = textstat.flesch_reading_ease(text)
                    # Calculate Flesch-Kincaid Grade Level
                    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)

                    # Append the results to the DataFrame
                    df.at[index, f'{rewrite}_Flesch_Reading_Ease'] = flesch_reading_ease
                    df.at[index, f'{rewrite}_Flesch_Kincaid_Grade'] = flesch_kincaid_grade

        # Concatenate Flesch_Reading_Ease and Flesch_Kincaid_Grade scores from all rewrites into single series
        flesch_reading_ease_series = pd.concat([df[col][df[col] > 0] for col in df.columns if '_Flesch_Reading_Ease' in col], ignore_index=True)
        flesch_kincaid_grade_series = pd.concat([df[col][df[col] > 0] for col in df.columns if '_Flesch_Kincaid_Grade' in col], ignore_index=True)
        self.final_data["flesch_reading_ease_series"] = flesch_reading_ease_series
        self.final_data["flesch_kincaid_grade_series"] = flesch_kincaid_grade_series

        # Define function for five-number summary
        def five_number_summary(series):
            return {
                "Min": series.min(),
                "Q1": series.quantile(0.25),
                "Median": series.median(),
                "Q3": series.quantile(0.75),
                "Max": series.max()
            }

        # Calculate five-number summary for each score
        reading_ease_summary = five_number_summary(flesch_reading_ease_series)
        grade_level_summary = five_number_summary(flesch_kincaid_grade_series)

        # Print summaries
        print("Flesch Reading Ease Summary:", reading_ease_summary)
        print("Flesch-Kincaid Grade Level Summary:", grade_level_summary)

        # Return the updated DataFrame with appended scores and summaries
        return df, reading_ease_summary, grade_level_summary
    
    def plot_flesch_histograms(self, Flesch_Reading_Ease, Flesch_Kincaid_Grade, save_path=None):
        # Font sizes for the plot
        title_fontsize = 20
        label_fontsize = 16
        tick_labelsize = 14  # Size of the tick labels

        # Create subplots (two panels in one figure)
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))

        # Plot for Flesch Reading Ease (Panel a)
        axs[0].hist(Flesch_Reading_Ease, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        axs[0].set_title('Flesch Reading Ease Distribution', fontsize=title_fontsize)
        axs[0].set_xlabel('Flesch Reading Ease', fontsize=label_fontsize)
        axs[0].set_ylabel('Frequency', fontsize=label_fontsize)
        axs[0].tick_params(axis='x', labelsize=tick_labelsize)  # Set x-axis tick label size
        axs[0].tick_params(axis='y', labelsize=tick_labelsize)
        axs[0].grid(True, alpha=0.4)

        # Add "(a)" in bold at the upper left corner of the first subplot
        axs[0].text(-0.1, 1.05, '(a)', fontsize=25, fontweight='bold', ha='center', va='center', transform=axs[0].transAxes)

        # Plot for Flesch-Kincaid Grade Level (Panel b)
        axs[1].hist(Flesch_Kincaid_Grade, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        axs[1].set_title('Flesch-Kincaid Grade Level Distribution', fontsize=title_fontsize)
        axs[1].set_xlabel('Flesch-Kincaid Grade Level', fontsize=label_fontsize)
        axs[1].set_ylabel('Frequency', fontsize=label_fontsize)
        axs[1].tick_params(axis='x', labelsize=tick_labelsize)  # Set x-axis tick label size
        axs[1].tick_params(axis='y', labelsize=tick_labelsize)
        axs[1].grid(True, alpha=0.4)

        # Add "(b)" in bold at the upper left corner of the second subplot
        axs[1].text(-0.1, 1.05, '(b)', fontsize=25, fontweight='bold', ha='center', va='center', transform=axs[1].transAxes)

        # Adjust layout and optionally save the figure
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    
    def orchestrator(self):
        filename = f"{self.data_path}/Readability_Results/Readability_Evaluation_Paraphrase.csv"
        self.run_model_for_flags(filename)
        
        
    def evaluator(self):
        ## Print five number summary scores:
        self.final_data["Readability_Evaluation_df"] = pd.read_csv(f"{self.data_path}/Readability_Results/Readability_Evaluation_Paraphrase.csv")
        self.calculate_and_append_scores(self.final_data["Readability_Evaluation_df"])
        ## Plot the figures 
        filename = f"{self.data_path}/Readability_Results/Readability_Plot.png"
        self.plot_flesch_histograms(self.final_data["flesch_reading_ease_series"], self.final_data["flesch_kincaid_grade_series"], filename)
            
            