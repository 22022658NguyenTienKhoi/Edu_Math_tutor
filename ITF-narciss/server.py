# server.py
import os
import uuid
import random
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Gemini & LangGraph Imports ---
from google import genai
from google.genai import types
from langgraph.graph import StateGraph

# --- Custom Module Imports ---
# Ensure these files are in your python path or same folder
from state_definitions import GraphState 
from error_detection_analysis import create_error_analysis_agent
from graph_conversation import create_tutoring_graph
from verify_presenter import create_presenter_agent # We will integrate the logic directly below to ensure dynamic state usage

load_dotenv()
# ==========================================
# 3. INITIALIZE SERVER & AGENTS
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agents ONCE
error_analysis_agent = create_error_analysis_agent()
tutoring_graph = create_tutoring_graph()
presenter_agent = create_presenter_agent()

# Load Data
try:
    DF = pd.read_csv('final_dataset_with_topic.csv')
    print(f"Data Loaded: {len(DF)} records.")
except:
    print("Warning: CSV not found, using mock data.")
    DF = pd.DataFrame()

# Global Session Store (The "Section" State)
SESSIONS = {}

# ==========================================
# 4. API MODELS & ENDPOINTS
# ==========================================

class InitRequest(BaseModel):
    topic: str

class SubmitRequest(BaseModel):
    session_id: str
    student_solution: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SessionRequest(BaseModel):
    session_id: str

@app.get("/")
async def root():
    return {"status": "MathMaster API Running"}

@app.post("/start")
async def start_exercise(req: InitRequest):
    """
    Selects a random question filtered by the requested topic.
    """
    session_id = str(uuid.uuid4())
    
    # Check if main Dataframe is loaded
    if DF.empty:
        raise HTTPException(status_code=500, detail="Database is empty")

    # Filter DF by the requested topic
    # We strip whitespace and lowercase just in case of formatting mismatch
    print(f"Requested topic: {req.topic}")
    topic_df = DF[DF['topic'] == req.topic]

    if topic_df.empty:
        # If no questions found for this topic, return 404
        raise HTTPException(status_code=404, detail=f"No questions found for topic: {req.topic}")

    # Sample from the filtered subset
    #row = topic_df.sample(1).iloc[0]
    row = DF.iloc[297]
    q_text = row['question']
    q_meta = row.to_dict()

    # Initialize GraphState
    initial_state = {
        "student_id": "Real_Web_User",
        "question_id": str(uuid.uuid4()),
        "question": q_text,
        "initial_student_solution": "",
        "student_input": "",
        "round": 0,
        "conversation_history": [],
        "is_correct": False,
        "is_final": False,
        "instructional_context": {},
        "error_analysis": {}, 
        "presentation": "",   
        "metadata": q_meta
    }
    
    # Save to memory
    SESSIONS[session_id] = initial_state
    
    return {
        "session_id": session_id,
        "question": q_text
    }

@app.post("/submit")
async def submit_solution(req: SubmitRequest):
    """
    Pipeline: Input -> Error Analysis Agent -> Presenter Agent -> Frontend
    """
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session expired")
    
    state = SESSIONS[req.session_id]
    
    # 1. Update State with Input
    state["initial_student_solution"] = req.student_solution
    state["student_input"] = req.student_solution
    
    try:
        # 2. Run Error Analysis Agent
        # This updates state['error_analysis'] and state['is_correct']
        print("--- Invoking Error Analysis ---")
        state = error_analysis_agent.invoke(state)
        
        # 3. Run Presenter Agent
        # This reads state['error_analysis'] and generates state['presentation']
        print("--- Invoking Presenter ---")
        state = presenter_agent.invoke(state)
        
        # 4. Save updated state back to global store
        SESSIONS[req.session_id] = state
        
        # 5. Determine status for Frontend UI colors
        # (Assuming error_analysis agent sets 'is_correct' boolean, or we parse it)
        is_correct = state.get("is_correct", False)
        
        # If the agent didn't set boolean, try to guess from classification string
        if not isinstance(is_correct, bool):
             err_data = state.get("error_analysis", {})
             if isinstance(err_data, dict):
                 is_correct = (err_data.get("classification") == "Correct")
        print(f"Determined is_correct: {is_correct}")
        print(f"Presentation: {state.get('presentation', '')}")
        return {
            "is_correct": is_correct,
            "title": "Assessment Result", 
            "message": state.get("presentation", "Analysis complete.")
        }
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(500, f"AI Processing Error: {str(e)}")

@app.post("/chat/start")
async def start_chat_session(req: SessionRequest):
    """
    Called when the student clicks 'Open Chat'. 
    The AI looks at the current error state and generates a greeting/hint.
    """
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session expired")
    
    state = SESSIONS[req.session_id]
    
    # We inject a hidden system prompt as the 'student_input'.
    # This guides the AI to start the conversation contextually.
    # The graph_conversation logic will process this as the context.
    state["student_input"] = (
        "System Notification: The student has requested help after a failed attempt. "
        "Review the 'error_analysis' in the state. "
        "Greet the student and provide a scaffolding question or hint based on their specific mistake. "
        "Do not give the answer."
    )
    
    try:
        # Run the Tutoring Graph
        output_state = tutoring_graph.invoke(state)
        
        # Update Session
        SESSIONS[req.session_id] = output_state
        
        # Get the AI's response (the greeting)
        last_msg = output_state['conversation_history'][-1]
        print(last_msg)
        return {
            "sender": "ai",
            "message": last_msg['content']['response']
        }
    except Exception as e:
        print(f"Chat Start Error: {e}")
        # Fallback if AI fails
        return {
            "sender": "ai", 
            "message": "Hello! I see you're working on this problem. How can I help you?"
        }

@app.post("/chat")
async def chat_with_tutor(req: ChatRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(404, "Session expired")
    
    state = SESSIONS[req.session_id]
    
    # Update input
    state["student_input"] = req.message
    
    try:
        # Run Tutoring Graph
        # This handles conversation history appending internally
        output_state = tutoring_graph.invoke(state)
        
        # Update Session
        SESSIONS[req.session_id] = output_state
        
        # Get latest AI response
        last_msg = output_state['conversation_history'][-1]
        
        return {
            "sender": "ai",
            "message": last_msg['content']['response']
        }
    except Exception as e:
        raise HTTPException(500, f"Chat Error: {str(e)}")

# Run command: uvicorn server:app --reload --host 0.0.0.0 --port 8000