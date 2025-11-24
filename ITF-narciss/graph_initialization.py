# /thay_tich_hop/state_definitions.py
# --- SETUP GEMINI (Same as before) ---
from dotenv import load_dotenv
from google.genai import types
from google import genai
from state_definitions import GraphState
from langgraph.graph import StateGraph
import json
import os
from utils import read_json
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
    "task_type": ['fileSearchStores/math10knowledgeandskills-b1dj50mjil84'],
    "textbook": ["fileSearchStores/math10knowledgeandskillstex-49spwnbo6cba", 'fileSearchStores/math10knowledgeandskillstex-a0lvhseev71i']
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
    question_id = state["question_id"]
    
    # Mocking data retrieval logic
    question_data = read_json("data/question_bank.json").get(question_id, {})
    concept_id = question_data.get("related_concept_id")
    task_type = question_data.get("task_type")
    textbook_content = read_json("data/textbook_content.json").get(concept_id, {})
    task_info = read_json("data/task_types.json").get(task_type, {})

    context = {
        "question_data": question_data,
        "related_definitions": textbook_content,
        "task_info": task_info
    }
    return {"instructional_context": context}

def error_detector_node(state: GraphState) -> dict:
    """
    Outputs string directly to 'error_analysis'.
    """
    print("---(2) Running Error Detector Node---")
    context = state["instructional_context"]
    student_solution = state["initial_student_solution"]
    
    prompt = f"""
    ANALYZE THIS STUDENT ANSWER.
    Question: {context.get("question_data", {}).get("question_text")}
    Student Answer: "{student_solution}"
    
    Task: Explain if the answer is correct or incorrect. If incorrect, identify the logic error type and explain why based on Logic theory.
    Output Format: Short concrete plain text paragraph around 50 words.
    """
    print(prompt)
    # Get text result
    analysis_text = call_gemini_model(prompt)
    
    # Return directly as string (matches GraphState type)
    return {"error_analysis": analysis_text}

def learner_modeller_node(state: GraphState) -> dict:
    """
    Outputs string directly to 'learner_profile'.
    """
    print("---(3) Running Learner Modeller Node---")
    student_id = state['student_id']
    
    # Access string directly
    analysis_text = state['error_analysis'] 
    context  = state["instructional_context"]
    prompt = f"""    
    CREATE A LEARNER PROFILE UPDATE.
    Student ID: {student_id}
    Recent Error Analysis: "{analysis_text}"
    
    Task: 
    - Retrieve knowledge from the task_type and solving_strategies for the question: {context.get("question_data", {}).get("question_text")}.
    - Write a short dynamic assessment based on this knowledge and recent error analysis. The assessment contains:
        - Learner’s representation of standards, competencies, and task requirements: learners understanding and representation of task requirements and the related competencies
        - Learner’s prior level of competencies (i.e., knowledge, meta-cognitive knowledge and strategies). 
    Output Format: Short concrete plain text paragraph around 50 words.
    """
    
    assessment_text = call_gemini_model(prompt, knowledge="task_type")
    
    # Return directly as string (matches GraphState type)
    return {"learner_profile": assessment_text}

def prompt_crafter_node(state: GraphState) -> dict:
    """
    Uses string inputs to create the Master Prompt.
    """
    print("---(4) Running Prompt Crafter Node---")
    
    # Direct string access
    analysis_text = state["error_analysis"]
    profile_text = state["learner_profile"]
    #initial_solution = state["initial_student_solution"]

    meta_prompt = f"""
    YOU ARE AN AI INSTRUCTIONAL DESIGNER.
    Create a SYSTEM PROMPT for a Tutor AI.

    Inputs:
    - Student Error: {analysis_text}
    - Student Profile: {profile_text}

    Your task: 
    - Retrieve the knowledge from the textbook content of menh de tap hop
    - Write a system prompt that defines the Tutor's persona and pedagogical strategy (scaffolding) for the next conversation turn.
    """

    system_prompt_text = call_gemini_model(meta_prompt, knowledge="textbook") 
    
    '''history = [
        {'role': 'system', 'content': system_prompt_text},
        {'role': 'user', 'content': initial_solution}
    ]'''
    
    return {"master_prompt": system_prompt_text}

# --- WORKFLOW SETUP ---

def create_error_analysis_agent():
    workflow = StateGraph(GraphState)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("error_detector", error_detector_node)
    workflow.add_node("learner_modeller", learner_modeller_node)
    workflow.add_node("prompt_crafter", prompt_crafter_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "error_detector")
    workflow.add_edge("error_detector", "learner_modeller")
    workflow.add_edge("learner_modeller", "prompt_crafter")
    workflow.set_finish_point("prompt_crafter")
    
    return workflow.compile()

if __name__ == '__main__':
    agent = create_error_analysis_agent()
    
    inputs = {
        "question_id": "QB_01_01_01",
        "student_id": "HS_123",
        "initial_student_solution": "Mệnh đề đảo: 'Nếu tam giác ABC không cân thì tam giác ABC không đều'.",
        "round": 0,
        "conversation_history": []
    }
    
    print("\n--- STARTING FLOW ---")
    result = agent.invoke(inputs)
    
    print("\n--- FINAL RESULTS (Text Fields) ---")
    print(f"[Error Analysis]:\n{result['error_analysis']}\n")
    print(f"[Learner Profile]:\n{result['learner_profile']}\n")
    print(f"[Master Prompt]:\n{result['master_prompt']}")