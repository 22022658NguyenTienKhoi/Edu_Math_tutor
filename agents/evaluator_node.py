# /educationq_project/agents/evaluator_node.py

import json
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from data_models import EducationQState, EvaluationReport, TestResult, QuestionItem, DialogueExchange
from typing import Dict, List, Any, Optional, TypedDict
class EvaluatorNode:
    """Handles evaluation: analyzes pre/post tests and dialogue."""

    def __init__(self, model_name, temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        #ChatOpenAI(model=model_name, temperature=temperature)

    def evaluate(self, state: EducationQState) -> EducationQState:
        """Evaluate the learning process and generate structured report."""
        print("📊 Evaluating learning process...")

        # Calculate basic metrics
        pre_accuracy = self._calculate_accuracy(state["pre_test"])
        post_accuracy = self._calculate_accuracy(state["post_test"])
        learning_gain = post_accuracy - pre_accuracy

        # Calculate additional metrics
        pnir = self._calculate_pnir(state["pre_test"], state["post_test"])
        css = self._calculate_css(state["pre_test"], state["post_test"])
        uic = self._calculate_uic(state["pre_test"], state["post_test"])

        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            state["questions"],
            state["pre_test"],
            state["post_test"],
            state["dialogue_history"]
        )

        # Create evaluation report
        evaluation = EvaluationReport(
            pre_test_accuracy=pre_accuracy,
            post_test_accuracy=post_accuracy,
            learning_gain=learning_gain,
            positive_negative_improvement_ratio=pnir,
            consistency_score=css,
            unique_improvement_count=uic,
            detailed_analysis=detailed_analysis
        )

        state["evaluation"] = evaluation

        print(f"✅ Evaluation completed:")
        print(f"   Pre-test accuracy: {pre_accuracy:.2%}")
        print(f"   Post-test accuracy: {post_accuracy:.2%}")
        print(f"   Learning gain (ALG): {learning_gain:.2%}")
        print(f"   PNIR: {pnir:.2f}")
        print(f"   CSS: {css:.2f}")
        print(f"   UIC: {uic}")

        return state

    def _calculate_accuracy(self, test_results: List[TestResult]) -> float:
        """Calculate accuracy from test results."""
        if not test_results:
            return 0.0
        correct = sum(1 for result in test_results if result.is_correct)
        return correct / len(test_results)

    def _calculate_pnir(self, pre_test: List[TestResult], post_test: List[TestResult]) -> float:
        """Calculate Positive-Negative Improvement Ratio."""
        if len(pre_test) != len(post_test):
            return 0.0

        improvements = []
        for pre, post in zip(pre_test, post_test):
            if pre.question_id == post.question_id:
                if not pre.is_correct and post.is_correct:
                    improvements.append(1)  # Positive improvement
                elif pre.is_correct and not post.is_correct:
                    improvements.append(-1)  # Negative improvement
                else:
                    improvements.append(0)  # No change

        if not improvements:
            return 0.0

        positive_count = sum(1 for imp in improvements if imp > 0)
        negative_count = sum(1 for imp in improvements if imp < 0)

        if negative_count == 0:
            return float('inf') if positive_count > 0 else 0.0

        return positive_count / negative_count

    def _calculate_css(self, pre_test: List[TestResult], post_test: List[TestResult]) -> float:
        """Calculate Consistency Score (standard deviation of gains)."""
        if len(pre_test) != len(post_test):
            return 0.0

        gains = []
        for pre, post in zip(pre_test, post_test):
            if pre.question_id == post.question_id:
                pre_score = 1 if pre.is_correct else 0
                post_score = 1 if post.is_correct else 0
                gains.append(post_score - pre_score)

        if len(gains) < 2:
            return 0.0

        return np.std(gains)

    def _calculate_uic(self, pre_test: List[TestResult], post_test: List[TestResult]) -> int:
        """Calculate Unique Improvement Count."""
        if len(pre_test) != len(post_test):
            return 0

        improvements = 0
        for pre, post in zip(pre_test, post_test):
            if pre.question_id == post.question_id:
                if not pre.is_correct and post.is_correct:
                    improvements += 1

        return improvements

    def _generate_detailed_analysis(self, questions: List[QuestionItem],
                                  pre_test: List[TestResult],
                                  post_test: List[TestResult],
                                  dialogue_history: List[DialogueExchange]) -> Dict[str, Any]:
        """Generate detailed analysis using LLM."""

        # Prepare data for analysis
        question_analysis = []
        for q, pre, post in zip(questions, pre_test, post_test):
            question_analysis.append({
                "question": q.question,
                "pre_correct": pre.is_correct,
                "post_correct": post.is_correct,
                "improved": not pre.is_correct and post.is_correct
            })

        dialogue_summary = "\n".join([
            f"Round {ex.round}: {ex.teacher_question}\nStudent: {ex.student_response}"
            for ex in dialogue_history
        ])

        prompt = f"""
        Analyze this educational assessment data and provide insights:

        Question Performance:
        {json.dumps(question_analysis, indent=2)}

        Dialogue Summary:
        {dialogue_summary}

        Provide a structured analysis including:
        1. Learning patterns observed
        2. Effectiveness of dialogue approach
        3. Areas of improvement
        4. Recommendations for future learning
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return {
            "llm_analysis": response.content,
            "question_breakdown": question_analysis,
            "dialogue_rounds": len(dialogue_history)
        }