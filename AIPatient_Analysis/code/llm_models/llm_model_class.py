from config import config
from openai import AzureOpenAI

class LLM_Models:
    def __init__(self):
            """
            Initialize the processor using configuration from the config file.
            :param secret_file: Path to the file containing the AzureOpenAI API key.
            """
            # Load the configuration from config.py
            self.raw_data_paths = config["raw_data_paths"]
            self.model = config["model"]
            self.max_tokens = config["max_tokens"]
            self.temperature = config["temperature"]
            """Initialize the class with the df_patient and store the patient pairs."""
            self.raw_data_paths = config["data_path"]
            self.secrete_file = config["secret_file"]
            # Load AzureOpenAI API key from the secret file
            self.openai_client = self._load_azure_client(self.secrete_file)
    
    def _load_azure_client(self, secret_file):
        """
        Load the AzureOpenAI client using the API key from the secret file.
        :param secret_file: Path to the file containing the AzureOpenAI API key and other configurations.
        :return: An initialized AzureOpenAI client.
        """
        with open(secret_file) as f:
            lines = f.readlines()
            config = {}
            for line in lines:
                key, value = line.split(',')
                config[key.strip()] = value.strip()

        azure_client = AzureOpenAI(
            api_key=config['azure_api_key'],
            api_version=config['api_version'],
            azure_endpoint=config['azure_endpoint'],
            organization=config['organization']
        )
        return azure_client

    def run_gpt(client, text_prompt, max_tokens_to_sample=3000, temperature=0, model='gpt-4-turbo'):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": text_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens_to_sample
        )
        return response.choices[0].message.content