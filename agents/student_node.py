# /educationq_project/agents/student_node.py

from typing import List
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from data_models import EducationQState, QuestionItem, TestResult, DialogueExchange

class StudentNode:
    """Handles student interactions: pre-test, dialogue, and post-test."""

    def __init__(self, model_name, temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        #(model=model_name, temperature=temperature)

    def pre_test(self, state: EducationQState) -> EducationQState:
        """Student takes initial test."""
        print("🎓 Student taking pre-test...")

        pre_test_results = []
        for question in state["questions"]:
            # Simulate student answering (in real implementation, this would be interactive)
            response = self._simulate_student_answer(question,state)
            pre_test_results.append(response)

        state["pre_test"] = pre_test_results
        state["round"] = 0
        print(f"✅ Pre-test completed. Accuracy: {self._calculate_accuracy(pre_test_results):.2%}")

        return state

    def dialogue(self, state: EducationQState) -> EducationQState:
        """Student responds to teacher's probing question."""
        if state["round"] > state["max_rounds"]:
            return state

        print(f"🎓 Student responding to round {state['round']}...")

        # Get the last teacher question from dialogue history
        if state["dialogue_history"]:
            last_exchange = state["dialogue_history"][-1]
            teacher_question = last_exchange.teacher_question
        else:
            teacher_question = "Let's discuss this question together."

        # Generate student response
        student_response = self._generate_dialogue_response(
            state["questions"],
            state["pre_test"],
            teacher_question,
            state["dialogue_history"]
        )

        # Create dialogue exchange
        exchange = DialogueExchange(
            round=state["round"],
            teacher_question=teacher_question,
            student_response=student_response
        )

        state["dialogue_history"].append(exchange)
        print(f"💬 Student: {student_response}...")

        return state

    def post_test(self, state: EducationQState) -> EducationQState:
        """Student retakes the test after dialogue."""
        print("🎓 Student taking post-test...")

        post_test_results = []
        for question in state["questions"]:
            # Student answers again, potentially with improved understanding
            response = self._simulate_student_answer(question, state, improved=True)
            post_test_results.append(response)

        state["post_test"] = post_test_results
        print(f"✅ Post-test completed. Accuracy: {self._calculate_accuracy(post_test_results):.2%}")

        return state

    def _simulate_student_answer(self, question: QuestionItem, state, improved: bool = False) -> TestResult:
        """Simulate student answering a question."""
        # In a real implementation, this would be interactive
        # For simulation, we'll use the LLM to generate answers
        context = ""
        if state['dialogue_history']:
            context = "\n".join([
                f"Round {ex.round}: Teacher: {ex.teacher_question}\nMe: {ex.student_response}"
                for ex in state['dialogue_history']
            ])

        pre_test_prompt = ChatPromptTemplate.from_template(
            """{context_msg}
        Answer this question by selecting the best option (0-{max_index}):

        Question: {question_text}
        Options:
        {options}
        The output must be Vietnamese.
        Try to think and then give final response with a JSON object: {{"reasoning": <try to solve the question first, once get the correct answer, create a wrong one >,
        "answer": <wrong final answer index chosen in reasoning>,
        "student_work":<full wrong solution of student for chosen answer. It must be student like>}}.
        "consistency": "The student academic traits and behaviors that make up the student work"
        """
        )
        post_test_prompt = ChatPromptTemplate.from_template(
            """{context_msg}
        Answer this question by selecting the best option (0-{max_index}):

        Question: {question_text}
        Options:
        {options}
        The output must be Vietnamese.
        Try to think and then give final response with a JSON object: {{"reasoning": <the student inner reasoning trace to solve problem>,
        "answer": <final answer index chosen in reasoning>,
        "student_work":<full solution of student for chosen answer. It must be student like>}}.
        """
        )
        json_parser = JsonOutputParser()

        if improved:
            prompt = post_test_prompt
        else:
            prompt = pre_test_prompt
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
        print(response)
        try:
            selected_answer = int(response["answer"])
            is_correct = selected_answer == question.correct_answer
        except (ValueError, KeyError, IndexError):
            selected_answer = 0
            is_correct = False

        return TestResult(
            question_id=question.id,
            selected_answer=selected_answer,
            is_correct=is_correct,
            student_explaination=response["student_work"],
            confidence=np.random.uniform(0.3, 0.9),  # Simulated confidence
            consistency=response.get("consistency", "N/A")
        )

    def _generate_dialogue_response(self, questions: List[QuestionItem],
                                  pre_test: List[TestResult],
                                  teacher_question: str,
                                  dialogue_history: List[DialogueExchange]) -> str:
        """Generate student's dialogue response."""

        # Build context from previous dialogue
        context = ""
        if dialogue_history:
            context = "\n".join([
                f"Round {ex.round}: Teacher: {ex.teacher_question}\nStudent: {ex.student_response}"
                for ex in dialogue_history[-3:]  # Last 3 exchanges for context
            ])
        consistency = pre_test[0].consistency if pre_test else "N/A"
        prompt = f"""
        You will be given question from a teacher agents. The point is to give answers that has consistency through the dialogue, also make hard cases for the teacher to probe.
        These are traits and behaviours {consistency} in a learning dialogue.
        The teacher has asked: "{teacher_question}"

        Previous context:
        dialogue history: {context}
        Questions and your pre-test answers:
        {', '.join([f'Q: {q.question} | Your answer: {pre_test[i].selected_answer}' for i, q in enumerate(questions)])}

        Respond naturally as a student would, showing your thinking process and any questions you have.
        Keep your response conversational and educational and in Vietnamese.
        You must stay loyal to your traits and behaviors.
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _calculate_accuracy(self, test_results: List[TestResult]) -> float:
        """Calculate accuracy from test results."""
        if not test_results:
            return 0.0
        correct = sum(1 for result in test_results if result.is_correct)
        return correct / len(test_results)