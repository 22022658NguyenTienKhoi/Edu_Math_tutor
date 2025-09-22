# /educationq_project/app.py

import gradio as gr
import uuid
import json
from pydantic import ValidationError

# Import the necessary components from your project files
from data_models import TeacherSessionState, QuestionItem, TestResult
from graph import get_graph

# --- App Initialization ---
try:
    graph = get_graph()
except Exception as e:
    import traceback
    traceback.print_exc()
    graph = None

# --- Gradio Event Handlers ---

def start_session(question_text, options_str, correct_answer_idx, student_explanation):
    """Initializes the state and invokes the graph for the first time."""
    if not graph:
        raise gr.Error("Graph could not be initialized. Check console for errors.")
    
    try:
        session_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": session_id}}

        initial_input = TeacherSessionState(
            session_id=session_id, student_id="web_user",
            question=QuestionItem(id="q1", question=question_text, options=[o.strip() for o in options_str.split(',')], correct_answer=correct_answer_idx),
            initial_student_answer=TestResult(question_id="q1", selected_answer=0, is_correct=False, student_explanation=student_explanation)
        )

        # Invoke the graph. It returns a TeacherSessionState object.
        paused_state_dict = graph.invoke(initial_input, config=thread_config)
        paused_state = TeacherSessionState(**paused_state_dict)
        # FIXED: Use dot notation to access attributes of the returned object
        first_question = paused_state.dialogue_history[-1].teacher_question
        chat_history = [[None, first_question]]
        
        # FIXED: Use the object's method to correctly create a JSON string
        state_json_str = paused_state.model_dump_json(indent=2)
        
        return thread_config, chat_history, state_json_str, gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        import traceback; traceback.print_exc(); raise gr.Error(f"Error: {e}")

def student_responds(student_message: str, chat_history: list, thread_config: dict):
    """Updates the paused state with the user's message and resumes the graph."""
    if not student_message.strip():
        # Just get the current state to display, but don't run the graph
        current_state_dict = graph.get_state(config=thread_config)
        current_state = TeacherSessionState(**current_state_dict.values)
        state_json_str = TeacherSessionState.model_validate(current_state).model_dump_json(indent=2)
        return chat_history, state_json_str, ""

    # 1. Get the current, paused state of the graph
    current_state_snapshot_dict = graph.get_state(config=thread_config)
    current_state_snapshot = TeacherSessionState(**current_state_snapshot_dict.values)
    # 2. Re-create the Pydantic object for safe manipulation
    current_state_obj = TeacherSessionState.model_validate(current_state_snapshot)

    # 3. FIXED: Update the dialogue history using dot notation on the object
    current_state_obj.dialogue_history[-1].student_response = student_message
    
    # 4. Update the graph's memory with the modified object
    graph.update_state(thread_config, current_state_obj)

    # 5. Resume the graph by calling invoke with None
    resumed_state_dict = graph.invoke(None, config=thread_config)
    resumed_state = TeacherSessionState(**resumed_state_dict)
    # 6. FIXED: Use dot notation to get the next question
    next_teacher_question = resumed_state.dialogue_history[-1].teacher_question
    chat_history.append([student_message, next_teacher_question])
    
    # 7. FIXED: Use the object's method for correct JSON serialization
    state_json_str = resumed_state.model_dump_json(indent=2)

    return chat_history, state_json_str, ""

# --- Gradio UI Layout (Unchanged) ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Math Tutor (LangGraph)") as demo:
    thread_state = gr.State()
    gr.Markdown("# 🤖 AI Math Tutor (Powered by LangGraph)")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Initial Problem Setup", open=True) as setup_box:
                q_input = gr.Textbox(label="Question", value="What is the value of 5 + 3 * 2?")
                q_options = gr.Textbox(label="Options (comma-separated)", value="16, 11, 13")
                q_correct_idx = gr.Number(label="Correct Option Index (from 0)", value=1, precision=0)
                q_student_exp = gr.Textbox(label="Student's Initial (Incorrect) Explanation", lines=3, value="I did 5 plus 3, which is 8, and then multiplied by 2 to get 16.")
                start_btn = gr.Button("Start Tutoring Session", variant="primary")
            with gr.Accordion("🔍 Agent's Internal State (Live)", open=False):
                state_view = gr.JSON(label="Current Session State")
        with gr.Column(scale=2):
            with gr.Group(visible=False) as chat_box:
                chatbot = gr.Chatbot(label="Tutor Conversation", height=500, avatar_images=("./user.png", "./teacher.png"))
                student_txt = gr.Textbox(show_label=False, placeholder="Type your answer here and press Enter...")
                student_txt.submit(student_responds, [student_txt, chatbot, thread_state], [chatbot, state_view, student_txt])
    start_btn.click(start_session, inputs=[q_input, q_options, q_correct_idx, q_student_exp], outputs=[thread_state, chatbot, state_view, setup_box, chat_box])

if __name__ == "__main__":
    demo.launch()