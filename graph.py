
# /educationq_project/graph.py

import os
from typing import Literal, List, Dict

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data_models import TeacherSessionState, DialogueExchange, EvaluationResult
# Import the state definition from our data models
from data_models import TeacherSessionState, EvaluationResult, DialogueExchange, TestResult
import dotenv
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")
# --- Define Agent Actions as Nodes ---
# --- Define Pydantic Models for Parsers ---
# These tell the parser what structure to expect from the LLM.

class EvaluationParser(BaseModel):
    """Pydantic model for parsing the LLM's evaluation output."""
    summary: str
    errors: List[Dict[str, str]]

class PlanParser(BaseModel):
    """Pydantic model for parsing the LLM's teaching plan."""
    plan: List[str]

# --- Define Agent Actions as Nodes ---

class TeacherWorkflow:
    def __init__(self, model_name="models/gemma-3-27b-it"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)

        # --- Create Chains with Integrated Parsers ---

        # Chain for the initial, deep evaluation of student work
        self.evaluation_chain = ( PromptTemplate.from_template( """You are an expert math teacher. Analyze the student's initial work to find all errors. Return ONLY a valid JSON object with "summary" and "errors" keys. Student's Incorrect Explanation: "{student_explanation}" Output format: summary: str errors: List[Dict[str, str]] = Field(default_factory=list)""" ) | self.llm | JsonOutputParser(pydantic_object=EvaluationParser) ) 
        # Chain for creating/updating the teaching plan 
        self.plan_chain = ( PromptTemplate.from_template( """You are an Vietnamese math instructional designer. Create a concise, step-by-step teaching plan to help a student solve the problem with respect Vietnamese education system. Problem: "{question}" Initial Error: "{initial_error}" Conversation History: {history} Output should be like json like this: "plan": [ "Step 1: Read the question carefully before solving.", "Step 2: Identify the first operation.", "Step 3: Perform the multiplication.", "Step 4: Then perform the addition.", "Step 5: Check your result with the order of operations." ] ⚠️ Important: Output must be valid JSON with ONLY a list of strings, not objects.""" ) | self.llm | JsonOutputParser(pydantic_object=PlanParser) ) 
        # Chain for lightweight evaluation of a student's answer to a probing question
        self.probing_evaluation_chain = ( 
            PromptTemplate.from_template( 
                """You are a teacher assistant. 
                Does the student's answer correctly address the teacher's question with respect to the plan step? 
                Teacher's Question: "{teacher_question}" 
                Student's Answer: "{student_response}"
                Plan: "{plan}"
                Return a json that contains following fields""" 
                ) | self.llm | JsonOutputParser() )

    def initial_analysis(self, state: TeacherSessionState) -> dict:
        print("--- Node: initial_analysis ---")
        # FIXED: Use dot notation to access attributes of the state object
        evaluation_data = self.evaluation_chain.invoke(
            {"student_explanation": state.initial_student_answer.student_explanation}
        )
        evaluation = EvaluationResult(round=0, **evaluation_data)
        new_plan_data = self._create_plan(state, [])
        return {
            "evaluation_history": state.evaluation_history + [evaluation],
            "current_teaching_plan": new_plan_data,
            "current_plan_step": 0
        }

    def evaluate_response(self, state: TeacherSessionState) -> dict:
        print("--- Node: evaluate_response ---")
        # FIXED: Use dot notation
        last_exchange = state.dialogue_history[-1]
        response = self.probing_evaluation_chain.invoke(
            {"teacher_question": last_exchange.teacher_question, "student_response": last_exchange.student_response, "plan": state.current_teaching_plan[state.current_plan_step]}
        )
        is_correct = 'true' in response.lower()
        print("response:", response)
        print(f"Evaluation: {'Correct' if is_correct else 'Incorrect'}")
        return {"last_evaluation_correct": is_correct}

    def replan(self, state: TeacherSessionState) -> dict:
        print("--- Node: replan ---")
        # FIXED: Use dot notation
        history = [f"T: {ex.teacher_question} S: {ex.student_response}" for ex in state.dialogue_history]
        new_plan_data = self._create_plan(state, history)
        return {"current_teaching_plan": new_plan_data, "current_plan_step": 0}

    def generate_question(self, state: TeacherSessionState) -> dict:
        print("--- Node: generate_question ---")
        # FIXED: Use dot notation
        step = state.current_plan_step
        if step >= len(state.current_teaching_plan):
            question = "Excellent. How would you solve the original problem now?"
        else:
            guideline = state.current_teaching_plan[step]
            prompt = PromptTemplate.from_template("Your goal is: \"{guideline}\". Ask one simple question. The question must be in Vietnamese")
            chain = prompt | self.llm | StrOutputParser()
            question = chain.invoke({"guideline": guideline})
        exchange = DialogueExchange(round=state.round + 1, teacher_question=question, student_response="")
        return {"dialogue_history": state.dialogue_history + [exchange], "round": state.round + 1}

    def _create_plan(self, state, history) -> dict:
        # FIXED: Use dot notation
        return self.plan_chain.invoke(
            {"question": state.question.question, "initial_error": state.initial_student_answer.student_explanation, "history": history}
        )

def should_replan(state: TeacherSessionState) -> Literal["replan", "advance_plan"]:
    # FIXED: Use dot notation
    print(f"--- Edge: should_replan (Correct: {state.last_evaluation_correct}) ---")
    return "replan" if not state.last_evaluation_correct else "advance_plan"

def get_graph():
    workflow = TeacherWorkflow()
    graph_builder = StateGraph(TeacherSessionState)
    graph_builder.add_node("initial_analysis", workflow.initial_analysis)
    graph_builder.add_node("evaluate_response", workflow.evaluate_response)
    graph_builder.add_node("replan", workflow.replan)
    graph_builder.add_node("generate_question", workflow.generate_question)
    graph_builder.add_node("advance_plan", lambda state: {"current_plan_step": state.current_plan_step + 1})
    graph_builder.set_entry_point("initial_analysis")
    graph_builder.add_edge("initial_analysis", "generate_question")
    graph_builder.add_edge("replan", "generate_question")
    graph_builder.add_edge("advance_plan", "generate_question")
    graph_builder.add_edge("generate_question", "evaluate_response")
    graph_builder.add_conditional_edges("evaluate_response", should_replan, {"replan": "replan", "advance_plan": "advance_plan"})
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory, interrupt_before=["evaluate_response"])