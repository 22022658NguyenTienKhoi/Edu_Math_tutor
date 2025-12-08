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
    model="models/gemini-flash-lite-latest"
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit()

STORE_MAPPING = {
    "task_type": ['fileSearchStores/math10tasktypes-6ywhsgmh52fd'],
    "textbook": ['fileSearchStores/math10knowledgeandskills-ezvt0h7ud27n']
}

def call_gemini_model(prompt: str, knowledge=None) -> str:
    """
    Calls Gemini. If retrieval fails, it alerts you instead of crashing.
    """
    target_store = STORE_MAPPING.get(knowledge)
    
    config = types.GenerateContentConfig(
        temperature=1, # Lower temperature helps with factual retrieval
        response_mime_type='application/json'
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
        response_json = json.loads(response.text)
        return response_json
    except Exception as e:
        return f"Error: {str(e)}"

# --- UPDATED NODES ---

def presenter_node(state: GraphState) -> dict:
    """
    Outputs string directly to 'error_analysis'.
    """
    print("---(2) Running Presenter Node---")
    context = state.get("instructional_context", {})
    prompt = f"""
    Provide verification presentation for THIS STUDENT ANSWER.
    Question: {context.get('question_data', 'This is for demo testing')}
    Correct Answer: {context.get('correct_answer', 'This is for demo testing')}
    Student Answer: {state.get('student_input', 'This is for demo testing')}
    [Error Analysis]:
    {state.get('error_analysis', 'This is for demo testing')}
    If correct, give brief congratulation. 
    If incorrect:
        - Student error analysis indicate easy correct mistake: 
            - Highlight directly the step in the student answer, no need to add explanation.
        - Student error analysis indicate complicate mistake: 
            - Add inline error indicator directly into the student answer. Error indicator is in form of [Error here: ...].
    NEVER inform knowledge or hint to student by any mean.
    NEVER inform the correct answer to the student.
    Return presentation strategies for the student based on the analysis above that has following field:
    is_correct: bool
    presentation: string:
    For example:
    {{
        "is_correct": False,
        "presentation": "Tâm đường tròn I(-2;3).Véc-tơ chỉ phương của tiếp tuyến là $\overrightarrow{{IM}}=(1, -2)$ **(Error here: $\overrightarrow{{IM}}$ là VTPT)**  
        .Phương trình tổng quát là $2(x+1)+1(y-1)=0 \Leftrightarrow 2x+y+1=0$ **(Error here: Dùng sai VTPT)**"
    }}
    """
    print(prompt)
    # Get text result
    presentation_text = call_gemini_model(prompt)
    
    # Return directly as string (matches GraphState type)
    return {"presentation": presentation_text['presentation'], "is_correct": presentation_text['is_correct']}

def green(text):
    return f"\033[92m{text}\033[0m"

def red(text):
    return f"\033[91m{text}\033[0m"

def presenter_display_node(state: GraphState) -> dict:
    if state['presentation'] == 'RED':
        print(f"\n---(Presenter) Presentation Output ---\n{red(state['presentation'])}\n")
    elif state['presentation'] == 'GREEN':
        print(f"\n---(Presenter) Presentation Output ---\n{green(state['presentation'])}\n")
    else:
        print(f"\n---(Presenter) Presentation Output ---\n{red(state['presentation'])}\n")
# --- WORKFLOW SETUP ---

def create_presenter_agent():
    workflow = StateGraph(GraphState)
    workflow.add_node("presenter", presenter_node)
    workflow.add_node("presenter_display", presenter_display_node)

    workflow.set_entry_point("presenter")
    workflow.add_edge("presenter", "presenter_display")
    workflow.set_finish_point("presenter_display")
    
    return workflow.compile()

if __name__ == '__main__':
    agent = create_presenter_agent()
    
    inputs = {
        "question": "Demo",
        "student_id": "HS_123",
        "initial_student_solution": "Đây là phép giao, ta tìm phần chung.Phần chung là khoảng (0;2)",
        "round": 0,
        "conversation_history": []
    }
    
    print("\n--- STARTING FLOW ---")
    result = agent.invoke(inputs)