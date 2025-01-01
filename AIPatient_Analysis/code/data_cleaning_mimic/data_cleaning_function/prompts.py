## Identify additional symptom (other than chief complaints) from history of present illness.
def extract_symptom_prompt(text):
    prompt = """You are a biomedical and AI researcher.
    Please identify the symptoms from the following text, and add duration, frequency, or intensity if any. Don't include any adjectives or adverbs. If the relationship is negation, include a [N] after Negation.
    All the extracted information should be in the exact wording as the text; do not paraphrase.
    If there are multiple are identified, separate with ";"; if none are identified, output <N/A>
    Output_format: Enclose your output in the following format. For example:
    {'Answer': {'Symptom': 'polypectomies', 'Duration': 'a week', 'Frequency': 'once per week', 'Intensity': '<N/A>', 'Negation': '[N]'}, {'Symptom': 'headache', 'Duration': '30 days', 'Frequency': 'twice every day', 'Intensity': 'severe', 'Negation': '<N/A>'}} ;
    The text is:
    """
    result = prompt+"\n"+text
    return result

def format_history_prompt(text):
    prompt = """You are a biomedical and AI researcher.
    Please identify and format the past medical history from the following text. Don't include any adjectives or adverbs. If the relationship is negation, include a [N] tag.
    All the extracted information should be in the exact wording as the text; do not paraphrase.
    If there are multiple are identified, separate with ";"; if none are identified, output <N/A>
    Output_format: Enclose your output within <root><root> tags. For example:
    <medical history>{'Answer': polypectomies; asthma}<medical history>
    The text is:
    """
    result = prompt+"\n"+text
    return result

def extract_allergies(text):
    prompt = """You are a biomedical and AI researcher.
    Please format the allergies and adverse reactions from the following text. Don't include any adjectives or adverbs.
    If the text shows one medication name or a list of medication names, you should consider it or them as allergies.
    If there are no allergies / medications or patient explicitly stated they don't have any allergies, output <N/A>.
    If there are multiple are identified, separate with ";"; if none are identified, output <N/A>
    All the extracted information should be in the exact wording as the text; do not paraphrase.
    Output_format: Enclose your output within <root><root> tags. For example:
    <allergies>{'Answer': penicillins; percocet; morphine}<allergies>
    The text is:
    """
    result = prompt+"\n"+text
    return result


def extract_socialhistory(text):
    prompt = """You are a biomedical and AI researcher.
    Please identify and format the social history from the following text. Don't include any adjectives or adverbs, and use short phrases.
    All the extracted information should be in the exact wording as the text; do not paraphrase.
    If there are multiple are identified, separate with ";"; if none are identified, output <N/A>
    Output_format: Enclose your output within <root><root> tags. For example:
    <social history>{'Answer': no substance use; lives with his parents; used to work in the construction business}<social history>
    The text is:
    """
    result = prompt+"\n"+text
    return result


def extract_familyhistory(text):
    prompt = """You are a biomedical and AI researcher.
    Please identify and format the family medical history from the following text. Don't include any adjectives or adverbs. Identify the family member, and their corresponding medical history. Don't include any adjectives or adverbs.
    If there are multiple are identified, separate with ";"; if none are identified, output <N/A>
    All the extracted information should be in the exact wording as the text; do not paraphrase.
    # Output_format: Enclose your output within <root><root> tags. For example:
    # <family history>{'Answer': {'Family Member': 'Mother', 'Medical History': 'Alzheimer's Disease'}, {'Family Member': 'Grandmother', 'Medical History': 'Breast Cancer'}<family history>
    The text is:
    """
    result = prompt+"\n"+text
    return result
