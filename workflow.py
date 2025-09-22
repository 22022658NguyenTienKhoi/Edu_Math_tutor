# /educationq_project/workflow.py

from langgraph.graph import StateGraph, END
from typing import Dict, List, Any

# Import from our custom modules
from data_models import EducationQState, QuestionItem
from agents import StudentNode, TeacherNode, EvaluatorNode

class EducationQWorkflow:
    """Main workflow orchestrator using LangGraph."""

    def __init__(self, model_name, student_model_name, temperature: float = 0.0):
        self.student = StudentNode(student_model_name, temperature)
        self.teacher = TeacherNode(model_name, temperature)
        self.evaluator = EvaluatorNode(model_name, temperature)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(EducationQState)
        workflow.add_node("student_pre_test", self.student.pre_test)
        workflow.add_node("teacher_ask", self.teacher.ask)
        workflow.add_node("student_dialogue", self.student.dialogue)
        workflow.add_node("student_post_test", self.student.post_test)
        workflow.add_node("evaluator", self.evaluator.evaluate)
        
        workflow.set_entry_point("student_pre_test")
        workflow.add_edge("student_pre_test", "teacher_ask")
        workflow.add_edge("teacher_ask", "student_dialogue")
        workflow.add_conditional_edges(
            "student_dialogue",
            self._should_continue_dialogue,
            {"continue": "teacher_ask", "post_test": "student_post_test"}
        )
        workflow.add_edge("student_post_test", "evaluator")
        workflow.add_edge("evaluator", END)
        return workflow.compile()

    def _should_continue_dialogue(self, state: EducationQState) -> str:
        """Determine if dialogue should continue or move to post-test."""
        return "post_test" if state["round"] >= state["max_rounds"] else "continue"

    def run(self, questions: List[QuestionItem]) -> Dict[str, Any]:
        """Run the complete EducationQ workflow."""
        print("🚀 Starting EducationQ Assessment...")
        initial_state = EducationQState(
            questions=questions,
            pre_test=[],
            dialogue_history=[],
            round=0,
            post_test=[],
            evaluation=None,
            current_question_id=None,
            max_rounds=3
        )
        final_state = self.workflow.invoke(initial_state)
        print("🎉 EducationQ Assessment completed!")
        return {
            "final_state": final_state,
            "evaluation": final_state["evaluation"],
            "summary": {
                "total_questions": len(questions),
                "dialogue_rounds": len(final_state["dialogue_history"]),
                "pre_accuracy": final_state["evaluation"].pre_test_accuracy,
                "post_accuracy": final_state["evaluation"].post_test_accuracy,
                "learning_gain": final_state["evaluation"].learning_gain
            }
        }