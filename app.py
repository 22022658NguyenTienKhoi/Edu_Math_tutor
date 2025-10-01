# /educationq_project/app.py

import gradio as gr
import uuid
import json
from pydantic import ValidationError

# Import the necessary components from your project files
from data_models import TutoringSessionState
from graph import get_tutoring_graph
from error_detectors import get_error_agent

# --- App Initialization ---
try:
    graph = get_tutoring_graph()
    error_agent = get_error_agent()
except Exception as e:
    import traceback
    traceback.print_exc()
    graph = None
    error_agent = None

# --- Gradio Event Handlers ---

def start_session(question_text, student_explanation):
    """Initializes the state and invokes the graph for the first time."""
    if not graph:
        raise gr.Error("Graph could not be initialized. Check console for errors.")
    
    try:
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

        initial_input = {
        "session_id": session_id,
        "student_id": "student_123",
        "problem_statement": question_text,
        "initial_student_solution": student_explanation,
        "round": 0
    }

        # Invoke the graph. It returns a TutoringSessionState object.
        error_state = error_agent.invoke(initial_input, config=config)
        paused_state = graph.invoke(error_state, config=config)
        # FIXED: Use dot notation to access attributes of the returned object
        first_question = paused_state['dialogue_history'][-1].teacher_question
        chat_history = [[None, first_question]]
        
        return config, chat_history, paused_state, gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        import traceback; traceback.print_exc(); raise gr.Error(f"Error: {e}")

def student_responds(student_message: str, chat_history: list, thread_config: dict):
    """Updates the paused state with the user's message and resumes the graph."""
    if not student_message.strip():
        # Just get the current state to display, but don't run the graph
        current_state_dict = graph.get_state(config=thread_config)
        current_state = TutoringSessionState(**current_state_dict.values)
        state_json_str = TutoringSessionState.model_validate(current_state).model_dump_json(indent=2)
        return chat_history, state_json_str, ""
    print(student_message)
    # 1. Get the current, paused state of the graph
    current_state_snapshot_dict = graph.get_state(config=thread_config)
    current_state_snapshot = TutoringSessionState(**current_state_snapshot_dict.values)
    # 2. Re-create the Pydantic object for safe manipulation
    current_state_obj = TutoringSessionState.model_validate(current_state_snapshot)

    # 3. FIXED: Update the dialogue history using dot notation on the object
    current_state_obj.dialogue_history[-1].student_response = student_message
    
    # 4. Update the graph's memory with the modified object
    #graph.update_state(thread_config, current_state_obj)
    current_state_obj.round += 1
    # 5. Resume the graph by calling invoke with None
    resumed_state_dict = graph.invoke(current_state_obj, config=thread_config)
    #print(f"Graph resumed: {resumed_state_dict}")
    resumed_state = TutoringSessionState(**resumed_state_dict)
    # 6. FIXED: Use dot notation to get the next question
    next_teacher_question = resumed_state.dialogue_history[-1].teacher_question
    print(f"analysis: {resumed_state.latest_synthesizer_report}")
    print(f"Next question: {next_teacher_question}")
    chat_history.append([student_message, next_teacher_question])
    
    # 7. FIXED: Use the object's method for correct JSON serialization
    state_json_str = resumed_state.model_dump_json(indent=2)

    return chat_history, state_json_str, ""
import pandas as pd
df = pd.read_csv("Student_error.csv")
# --- Gradio UI Layout (Unchanged) ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Math Tutor (LangGraph)") as demo:
    thread_state = gr.State()
    gr.Markdown("# 🤖 AI Math Tutor (Powered by LangGraph)")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Initial Problem Setup", open=True) as setup_box:
                q_input = gr.Textbox(label="Question", value=df.iloc[-1]["Problem"])
                q_student_exp = gr.Textbox(label="Student's Initial (Incorrect) Explanation", lines=3, value=df.iloc[-1]["Student_solution"])
                start_btn = gr.Button("Start Tutoring Session", variant="primary")
            with gr.Accordion("🔍 Agent's Internal State (Live)", open=False):
                state_view = gr.JSON(label="Current Session State")
        with gr.Column(scale=2):
            with gr.Group(visible=False) as chat_box:
                chatbot = gr.Chatbot(label="Tutor Conversation", height=500, avatar_images=("./user.png", "./teacher.png"), type='tuples')
                student_txt = gr.Textbox(show_label=False, placeholder="Type your answer here and press Enter...")
                student_txt.submit(student_responds, [student_txt, chatbot, thread_state], [chatbot, state_view, student_txt])
    start_btn.click(start_session, inputs=[q_input, q_student_exp], outputs=[thread_state, chatbot, state_view, setup_box, chat_box])

if __name__ == "__main__":
    demo.launch()