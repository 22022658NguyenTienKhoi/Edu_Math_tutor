# /educationq_project/teacher_agent.py

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from typing import List
from data_models import TeacherSessionState, DialogueExchange, EvaluationResult, DeepAnalysisReport, TestResult

class TeacherNode:
    """A stateless agent that uses a multi-step plan to teach a concept."""
    def __init__(self, model_name: str, temperature: float = 0.0):
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY environment variable not set!")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, convert_system_message_to_human=True)

    # --- Step 1: Initial, Deep Analysis Functions ---

    def initial_analysis_and_plan(self, state: TeacherSessionState) -> TeacherSessionState:
        """Performs the initial deep analysis and creates the first teaching plan."""
        print("🕵️ Performing initial deep analysis...")
        # A) Evaluate the student's first attempt
        evaluation = self._evaluate_initial_work(state.initial_student_answer)
        state.evaluation_history.append(evaluation)

        # B) Create the first teaching plan based on that evaluation
        state.current_teaching_plan = self._update_teaching_plan(state)
        state.current_plan_step = 0
        print(f"📝 Initial plan created: {state.current_teaching_plan}")
        return state

    def _evaluate_initial_work(self, initial_answer: TestResult) -> EvaluationResult:
        """Performs a deep dive into the student's first explanation to find all errors."""
        prompt = f"""You are an expert Vietnamese math teacher. Analyze the student's initial work to identify all errors. Return ONLY a valid JSON object with "summary" and "errors" keys. Student's Incorrect Explanation: "{initial_answer.student_explanation}" """
        response_str = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return EvaluationResult(round=0, **self._safe_json_load(response_str))

    # --- Step 2: Conversational Turn Processing ---

    def process_student_response(self, state: TeacherSessionState) -> TeacherSessionState:
        """Evaluates a student's answer to a probing question and decides the next step."""
        if state.round >= state.max_rounds:
            return state

        last_exchange = state.dialogue_history[-1]
        teacher_question = last_exchange.teacher_question
        student_response = last_exchange.student_response

        print(f"🧑‍🎓 Student responded. Evaluating if the answer meets the goal...")
        is_correct = self._evaluate_probing_answer(teacher_question, student_response)

        if is_correct:
            print("✅ Correct! Moving to the next step in the plan.")
            state.current_plan_step += 1
        else:
            print("❌ Not quite. Revising the plan to address the confusion.")
            state.current_teaching_plan = self._update_teaching_plan(state) # Re-plan based on the new failure
            state.current_plan_step = 0 # Start from the top of the new, more detailed plan

        probing_question = self._generate_probing_question(state)
        
        # Update state for the next turn
        new_exchange = DialogueExchange(round=state.round + 1, teacher_question=probing_question, student_response="")
        state.dialogue_history.append(new_exchange)
        state.round += 1
        
        return state
    
    def _evaluate_probing_answer(self, teacher_question: str, student_response: str) -> bool:
        """A lightweight check to see if the student's answer correctly addresses the teacher's specific question."""
        prompt = f"""You are a teacher assistant. Your job is to determine if the student's answer correctly addresses the teacher's question.
        Teacher's Question: "{teacher_question}"
        Student's Answer: "{student_response}"
        Does the student's answer demonstrate understanding of the concept asked in the question?
        Respond with only the word 'true' or 'false'.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        return 'true' in response

    # --- Step 3: Shared Helper Functions ---

    def _update_teaching_plan(self, state: TeacherSessionState) -> List[str]:
        """Creates or revises the teaching plan based on all available history."""
        # The prompt is now more powerful as it can react to failures mid-conversation
        prompt = f"""You are an instructional designer. Create a concise, step-by-step teaching plan (as a JSON array of strings) to help a student solve: "{state.question.question}".
        Initial Error: "{state.initial_student_answer.student_explanation}"
        Full Conversation History: {[f'T: {ex.teacher_question} S: {ex.student_response}' for ex in state.dialogue_history]}
        Evaluation Summaries: {[e.summary for e in state.evaluation_history]}
        Based on this, what is the best sequence of pedagogical guidelines to help the student now? Start with the most fundamental step.
        """
        response_str = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return self._safe_json_load(response_str)

    def _generate_probing_question(self, state: TeacherSessionState) -> str:
        """Generates the next question based on the current step in the teaching plan."""
        # Check if the plan is completed
        if state.current_plan_step >= len(state.current_teaching_plan):
            return "Excellent! You seem to have a good grasp of that now. Let's try applying it. Can you solve the original problem again from start to finish?"

        current_guideline = state.current_teaching_plan[state.current_plan_step]
        prompt = f"""You are a friendly Vietnamese math teacher. Your current teaching goal is: "{current_guideline}".
        Based on this goal, formulate one clear, simple question to guide the student.
        Do not give the answer. Just ask the question.
        """
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

    def _safe_json_load(self, text: str):
        try: return json.loads(text)
        except json.JSONDecodeError: return {}