# /educationq_project/test_graph.py

import os
import uuid

from data_models import TeacherSessionState, QuestionItem, TestResult
from graph import get_graph

def test_case_1_initial_invocation(graph):
    print("\n--- Running Test Case 1: Initial Invocation ---")
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_input = TeacherSessionState(
        session_id="test_session_1",
        student_id="test_student",
        question=QuestionItem(id="q1", question="What is 5 + 3 * 2?", options=["16", "11"], correct_answer=1),
        initial_student_answer=TestResult(question_id="q1", selected_answer=0, is_correct=False, student_explanation="I did 5 plus 3 to get 8, then multiplied by 2 to get 16.")
    )
    print("Invoking graph...")

    # LangGraph's invoke method returns a TeacherSessionState object when the state is a Pydantic class
    paused_state_dict = graph.invoke(initial_input, config=thread_config)
    paused_state = TeacherSessionState(**paused_state_dict)
    print("Graph paused as expected.")
    
    # FIXED: Use dot notation for all assertions on the returned state object
    assert paused_state.round == 1, f"Expected round 1, got {paused_state.round}"
    assert len(paused_state.evaluation_history) == 1, "Initial evaluation should have been logged."
    assert len(paused_state.current_teaching_plan) > 0, "A teaching plan should have been created."
    assert len(paused_state.dialogue_history) == 1, "Dialogue history should have one entry."
    assert paused_state.dialogue_history[-1].student_response == "", "Student response slot should be empty."

    print("\n[SUCCESS] Test Case 1 Passed!")
    print(f"Initial Plan: {paused_state.current_teaching_plan}")
    print(f"First Question: {paused_state.dialogue_history[-1].teacher_question}")
    return graph, thread_config

def test_case_2_correct_answer(graph, thread_config):
    print("\n--- Running Test Case 2: Student Answers Correctly ---")
    # get_state returns a state object, not just a dict
    current_state_snapshot_dict = graph.get_state(config=thread_config)
    current_state_snapshot = TeacherSessionState(**current_state_snapshot_dict.values)
    #print(current_state_snapshot)
    correct_response = "Multiplication comes first, so 3*2 is 6, and 5+6 is 11."
    
    # FIXED: Update the object's attribute directly
    current_state_snapshot.dialogue_history[-1].student_response = correct_response
    print(current_state_snapshot)
    graph.update_state(thread_config, current_state_snapshot)
    print("State updated. Resuming graph...")

    resumed_state_dict = graph.invoke(None, config=thread_config)
    resumed_state = TeacherSessionState(**resumed_state_dict)
    print("Graph paused again.")

    # FIXED: Use dot notation for assertions
    assert resumed_state.round == 2
    assert resumed_state.last_evaluation_correct is True
    assert resumed_state.current_plan_step  == 1
    print("\n[SUCCESS] Test Case 2 Passed!")

def test_case_3_incorrect_answer(graph):
    print("\n--- Running Test Case 3: Student Answers Incorrectly ---")
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_input = TeacherSessionState(
        session_id="test_session_2",
        student_id="test_student_2",
        question=QuestionItem(id="q1", question="What is 5 + 3 * 2?", options=["16", "11"], correct_answer=1),
        initial_student_answer=TestResult(question_id="q1", selected_answer=0, is_correct=False, student_explanation="I just went left to right.")
    )
    
    paused_state: TeacherSessionState = graph.invoke(initial_input, config=thread_config)
    original_plan = paused_state.current_teaching_plan
    
    incorrect_response = "I don't get it."
    
    # FIXED: Update the object's attribute directly
    paused_state.dialogue_history[-1].student_response = incorrect_response
    
    graph.update_state(thread_config, paused_state) # Pass the full state object back
    print("State updated. Resuming graph...")

    resumed_state: TeacherSessionState = graph.invoke(None, config=thread_config)
    new_plan = resumed_state.current_teaching_plan
    
    # FIXED: Use dot notation for assertions
    assert resumed_state.last_evaluation_correct is False
    assert resumed_state.current_plan_step == 0
    assert original_plan != new_plan
    print("\n[SUCCESS] Test Case 3 Passed!")

if __name__ == "__main__":
    print("Initializing graph for testing...")
    main_graph = get_graph()
    graph_after_1, session_config = test_case_1_initial_invocation(main_graph)
    test_case_2_correct_answer(graph_after_1, session_config)
    test_case_3_incorrect_answer(main_graph)
    print("\n--- All test cases completed successfully! ---")