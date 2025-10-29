from error_detectors import ErrorAgent, get_error_agent
import pandas as pd
import json
import os
import glob, re

'''wrong_answer_dir = "//workspaces//Edu_Math_tutor//wrong_answer"

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

json_files = sorted(glob.glob(os.path.join(wrong_answer_dir, "question_*.json")), key=natural_key)

print(json_files)
data = []
for i in range(1, len(json_files)):
    with open(json_files[i], "r", encoding="utf-8") as f:
        data.append(json.load(f))
        print(f"Processing file: {json_files[i]}")

    # Step 1: Go inside wrong_solutions, keep meta
    df = pd.json_normalize(
        data,
        record_path=["wrong_solutions"],            # dive into wrong_solutions
        meta=["question"]                           # keep question for context
    )

    # Step 2: Explode applied_errors (list of dicts)
    df = df.explode("applied_errors")

    # Step 3: Flatten applied_errors into separate columns
    df = pd.concat(
        [df.drop(columns=["applied_errors"]),
         df["applied_errors"].apply(pd.Series)], 
        axis=1
    )

    # Reorder columns for clarity
    df = df[[
        "question", 
        "transformed_solution", 
        "wrong_step", 
        "error_type", 
        "description", 
        "is_single_error", 
        "notes"
    ]]'''
df = pd.read_csv("k10-renew.csv")
df["agent_analysis"] = None  
df = df.iloc[:100]
for index, row in df.iterrows():
    app = get_error_agent()
    config = {"configurable": {"thread_id": "session_001"}}
    initial_state = {
        "session_id": "session_001",
        "student_id": "student_123",
        "problem_statement": row["question"],
        "initial_student_solution": row['wrong_solution'],
        "round": 0
    }
    final_state = app.invoke(initial_state, config=config)
    latest_report = final_state['latest_synthesizer_report']
    df.at[index, 'agent_analysis'] = str(latest_report)  # Directly add to original df
    if index % 10 == 0:
        output_csv = f"new_test/student_error_{index}.csv"
        df.iloc[:index+1].to_csv(output_csv, index=False)
        print(f"Processed row {index}/{len(df)}")