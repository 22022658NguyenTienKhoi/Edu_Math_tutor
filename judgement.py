import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import glob, re
import pandas as pd
# --- Environment Setup ---
import dotenv
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

DEFAULT_REASONING_MODEL = "models/gemma-3-27b-it"
#models/gemini-2.5-flash
DEFAULT_FAST_MODEL = "models/gemma-3-27b-it"
wrong_answer_dir = "//workspaces//Edu_Math_tutor//test"

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

csv_files = sorted(glob.glob(os.path.join(wrong_answer_dir, "student_error_*.csv")), key=natural_key)
prompt_template ="""Evaluate the error_analysis {error_analysis} against the groud truth explanation {explanation}.
                        You return a json structure:
                        point: 1 if error_analysis match ground truth, 0 if not.
                        reason: explain why.
                     """
llm = ChatGoogleGenerativeAI(model=DEFAULT_REASONING_MODEL, temperature=0.2)
prompt = PromptTemplate.from_template(prompt_template)
chain = prompt | llm
for i in range(1, len(csv_files)):
    df = pd.read_csv(csv_files[i])
    df['res'] = None
    for index, row in df.iterrows():
        res = chain.invoke({"error_analysis": row['agent_analysis'], "explanation": row['explanation']})
        df.at[index, 'res'] = res
    df.to_csv(csv_files[i])
    print(f'saved to {csv_files[i]}')