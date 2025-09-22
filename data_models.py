# /educationq_project/data_models.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Base Data Items (Unchanged) ---

class QuestionItem(BaseModel):
    """Represents a single test question item."""
    id: str
    question: str
    options: List[str]
    correct_answer: int  # Index of correct option

class TestResult(BaseModel):
    """Represents a student's answer to a question."""
    question_id: str
    selected_answer: int
    is_correct: bool
    student_explanation: str

class DialogueExchange(BaseModel):
    """Represents a single round of conversation."""
    round: int
    teacher_question: str
    student_response: str
    timestamp: datetime = Field(default_factory=datetime.now)

# --- Teacher's Memory and Analysis Models (Unchanged) ---

class EvaluationResult(BaseModel):
    """Structured evaluation of a single student response."""
    round: int
    summary: str
    errors: List[Dict[str, str]] = Field(default_factory=list)

class DeepAnalysisReport(BaseModel):
    """In-depth analysis of recurring student issues."""
    round: int
    core_problem: str
    analysis_details: str
    recommended_focus: str

class TeacherSessionState(BaseModel):
    """Manages the entire state for a single-question learning session."""
    session_id: str
    student_id: str
    question: QuestionItem
    initial_student_answer: TestResult
    max_rounds: int = 10
    round: int = 0
    dialogue_history: List[DialogueExchange] = Field(default_factory=list)
    
    evaluation_history: List[EvaluationResult] = Field(default_factory=list)
    deep_analysis_reports: List[DeepAnalysisReport] = Field(default_factory=list)
    current_teaching_plan: List[str] = Field(default_factory=list)
    current_plan_step: int = 0
    
    last_evaluation_correct: bool = False