# /educationq_project/agents/student_node.py

from typing import List
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from data_models import EducationQState, QuestionItem, TestResult, DialogueExchange, StudentProfile

class StudentNode:
    """Handles student interactions: pre-test, dialogue, and post-test."""

    def __init__(self, model_name, temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        #(model=model_name, temperature=temperature)

    # personality from building graph.
    # model generate works of student. map to personality traits.
    # student profile: grade level, learning style, personality traits, student work.

    def self_reflection(self, state: EducationQState) -> EducationQState:
        
        print("🎓 Writer taking self-reflection...")

        '''pre_test_results = []
        for question in state["questions"]:
            # Simulate student answering (in real implementation, this would be interactive)
            response = self._simulate_student_answer(question,state)
            pre_test_results.append(response)

        state["pre_test"] = pre_test_results
        state["round"] = 0
        print(f"✅ Pre-test completed. Accuracy: {self._calculate_accuracy(pre_test_results):.2%}")'''
        # evaluate student profile for consistency
        return StudentProfile(
            grade_level="10th",
            learning_style="Visual",
            personality_traits=["Curious", "Diligent"],
            student_work="No work provided."
        )

    def _makeup_student_work(self, question: QuestionItem, improved: bool = False) -> StudentProfile:
        """Simulate student answering a question."""
        # In a real implementation, this would be interactive
        # For simulation, we'll use the LLM to generate answers


        '''prompt = ChatPromptTemplate.from_template(
            """{context_msg}
        You will play the role of a screen writer. You are given a question and answer. 
        You would create a wrong answer with a full solution that looks like a student work.
        Along with the student work is the reasoning process of the student to get to the wrong answer.
        Also, you must generate a set of student personality traits that would reflect in the student work.
        Question: {question_text}
        Options:
        {options}
        The output must be Vietnamese.
        Try to think and then give final response with a JSON object: {{"reasoning": <try to solve the question first, once get the correct answer, create a wrong one >,
        "answer": <wrong final answer index chosen in reasoning>,
        "student_work":<full wrong solution of student for chosen answer. It must be student like>}}.
        """
        )

        json_parser = JsonOutputParser()

        chain = prompt | self.llm | json_parser

        # Build inputs
        context_msg = (
            f"You are a Vietnamese student taking a test. Here is the previous conversation with your teacher\n{context}."
            if improved else
            "You are a k-10 Vietnamese student, this is your first attempt at this question, you must give the wrong answer"
        )

        inputs = {
            "context_msg": context_msg,
            "max_index": len(question.options) - 1,
            "question_text": question.question,
            "options": "\n".join([f"{i}. {opt}" for i, opt in enumerate(question.options)]),
        }
        print(inputs)
        # Run the chain
        response = chain.invoke(inputs)  # response is already parsed dict
        print(response)'''

        return StudentProfile(
            grade_level="10th",
            learning_style="Visual",
            personality_traits=["Curious", "Diligent"],
            student_work="No work provided."
        )