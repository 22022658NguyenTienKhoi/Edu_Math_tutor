from typing import TypedDict, Dict, Optional, List, Annotated
import operator

# --- 1. UPDATED GRAPH STATE (CLEANER) ---
class GraphState(TypedDict):
    """
    State definition optimized for Text-only flow.
    """
    # === Session Data ===
    student_id: str
    question_id: str
    initial_student_solution: str
    round: int

    # === Current Turn Data ===
    student_input: str

    # === Agent Outputs (Now Strings instead of Dicts) ===
    instructional_context: Optional[Dict] # Remains Dict (structured data from DB/File)
    error_analysis: Optional[str]         # CHANGED: Direct text analysis
    learner_profile: Optional[str]        # CHANGED: Direct text assessment
    master_prompt: Optional[str]          # Direct text prompt

    # === Conversation Management ===
    conversation_history: Annotated[list, operator.add]
    is_correct: bool