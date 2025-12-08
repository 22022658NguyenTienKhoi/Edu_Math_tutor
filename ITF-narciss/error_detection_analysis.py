# /thay_tich_hop/state_definitions.py
# --- SETUP GEMINI (Same as before) ---
from dotenv import load_dotenv
from google.genai import types
from google import genai
import ast
from state_definitions import GraphState
from langgraph.graph import StateGraph
import json
import os
from utils import read_json
import pandas as pd
import random
load_dotenv()

# Initialize Client
try:
    print("---(Setup) Initializing Gemini...---")
    client = genai.Client()
    model="models/gemini-flash-latest"
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit()

STORE_MAPPING = {
    "task_type": ['fileSearchStores/math10tasktypes-6ywhsgmh52fd'],
    "textbook": ['fileSearchStores/math10knowledge-xwv7tlbc7xyb']
}

def call_gemini_model(prompt: str, knowledge=None) -> str:
    """
    Calls Gemini. If retrieval fails, it alerts you instead of crashing.
    """
    target_store = STORE_MAPPING.get(knowledge)
    
    config = types.GenerateContentConfig(
        temperature=0, # Lower temperature helps with factual retrieval
    )

    if target_store:
        print(f"--- Attempting retrieval from store: {knowledge} ---")
        config.tools = [
            types.Tool(
                file_search=types.FileSearch(file_search_store_names=target_store)
            )
        ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        
        # --- RETRIEVAL CHECK LOGIC ---
        if target_store:
            # 1. Check if metadata exists (Did it even try?)
            if not response.candidates[0].grounding_metadata:
                print(f"WARNING: Model ignored the file store '{knowledge}' entirely. (No metadata)")
            
            # 2. Check if supports exist (Did it find anything?)
            elif not response.candidates[0].grounding_metadata.grounding_supports:
                print(f"WARNING: Model searched '{knowledge}' but found NO MATCHING content to cite.")
                # This usually means your prompt keywords don't match the text in the file.
            
            # 3. If successful, process data
            else:
                print("SUCCESS: Grounding data found.")
                for support in response.candidates[0].grounding_metadata.grounding_supports:
                    print(f"  Segment: {support.segment.text[:50]}...")
                    for chunk_index in support.grounding_chunk_indices:
                         # Defensive check for chunk index
                         chunks = response.candidates[0].grounding_metadata.grounding_chunks
                         if chunk_index < len(chunks):
                             chunk = chunks[chunk_index]
                             print(f"  -> Source: {chunk.retrieved_context.text[:100]}...")
        # -----------------------------

        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- UPDATED NODES ---

def retriever_node(state: GraphState) -> dict:
    """
    Retrieves structured context (Keeps Dict output).
    """
    print("---(1) Running Retriever Node---")
    question_df = pd.read_json('question/total_added.jsonl', lines=True)
    task_type_df = pd.read_json('task_type/total.json')
    input_solution_df = pd.read_csv('final_dataset_with_topic.csv')
    input_solution = state.get("initial_student_solution")
    #input_solution = ast.literal_eval(input_solution)
    #question_solution = input_solution_df.loc[input_solution_df['wrong_solution'] == input_solution].iloc[0]
    #print(question_solution)question_solution['question']
    question_text = state.get('question')
    question = question_df.loc[question_df['question'] == question_text].iloc[0]
    print(question['task_type'])
    task_type = task_type_df.loc[task_type_df['task type'] == question['task_type']].iloc[0]
    print(f'task_type{task_type}')
    prompt = f'''retrieve chunks relevant to the following question and solution.
    Question: {question['question']}
    Correct Answer: {question['solution']}
    Task_type: {question['task_type']}
    Task: retrieve relevant textbook content to support tutor on error analysis and teaching.
    If no relevant content found, respond with "NO RELEVANT CONTENT".
    '''
    chunks = call_gemini_model(prompt)
    context = {
        "question_data": question['question'],
        "correct_answer": question['solution'],
        "task_info": task_type,
        "retrieved_chunks": chunks
    }
    return {"instructional_context": context}

def error_detector_node(state: GraphState) -> dict:
    """
    Outputs string directly to 'error_analysis'.
    """
    print("---(2) Running Error Detector Node---")
    context = state["instructional_context"]
    student_solution = state["initial_student_solution"]
    #.get("question_text")}
    prompt = f"""
    ANALYZE THIS STUDENT ANSWER.
    Question: {context.get("question_data", {})}
    Correct Answer: {context.get("correct_answer", {})}
    Student Answer: "{student_solution}"
    Task_type: {context.get("task_info", {})}
    Relevant Textbook Chunks: {context.get("retrieved_chunks", {})}
    Task: 
    Verify correctness of the solution. If incorrect, identify the source of the error based on mathematical theory and explain why.
    Output Format: Short concrete plain text paragraph around 50 words.
    """
    print(prompt)
    # Get text result
    analysis_text = call_gemini_model(prompt)
    
    # Return directly as string (matches GraphState type)
    return {"error_analysis": analysis_text}


# --- WORKFLOW SETUP ---

def create_error_analysis_agent():
    workflow = StateGraph(GraphState)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("error_detector", error_detector_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "error_detector")
    workflow.set_finish_point("error_detector")
    
    return workflow.compile()

if __name__ == '__main__':
    agent = create_error_analysis_agent()
    input_solution_df = pd.read_csv('final_dataset_with_topic.csv')
    rand = random.randint(0, len(input_solution_df)-1)
    print(f"Selected input index: {rand}")
    input_solution = input_solution_df.iloc[rand]
    print(f"Selected wrong solution: {input_solution['wrong_solution']}")
    inputs = {
        "question_id": "QB_01_01_01",
        "student_id": "HS_123",
        "initial_student_solution": input_solution['wrong_solution'],
        "round": 0,
        "conversation_history": []
    }
    
    print("\n--- STARTING FLOW ---")
    result = agent.invoke(inputs)
    
    print("\n--- FINAL RESULTS (Text Fields) ---")
    print(f"[Error Analysis]:\n{result['error_analysis']}\n")