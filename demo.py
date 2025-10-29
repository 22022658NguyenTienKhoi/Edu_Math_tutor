from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END


# -------------------------
# Data contracts and state
# -------------------------

Mode = Literal["S1", "S2"]


class EpisodicTurn(TypedDict):
    role: Literal["student", "tutor"]
    content: str


class EpisodicMemory(TypedDict):
    turns: List[EpisodicTurn]


class CriticState(TypedDict, total=False):
    turns_seen: int
    current_mode: Mode
    consecutive_failures: int
    recent_misconceptions: List[str]
    last_strategy: Optional[str]
    last_confidence: float


class GraphState(TypedDict, total=False):
    student_response: str
    episodic: EpisodicMemory
    critic_state: CriticState
    feedback_or_plan: Optional[Dict[str, Any]]


# -------------------------
# Switching logic
# -------------------------

def update_switching_logic(critic: CriticState, progress_signal: str) -> CriticState:
    updated: CriticState = CriticState(**critic)
    updated["turns_seen"] = updated.get("turns_seen", 0) + 1

    if progress_signal == "improving":
        updated["consecutive_failures"] = 0
    else:
        updated["consecutive_failures"] = updated.get("consecutive_failures", 0) + 1

    if updated.get("current_mode", "S1") == "S1":
        if updated["turns_seen"] >= 5 or updated["consecutive_failures"] >= 3:
            updated["current_mode"] = "S2"
    else:
        if progress_signal == "improving":
            updated["current_mode"] = "S1"
        elif updated["consecutive_failures"] >= 2:
            pass

    return updated


# -------------------------
# Nodes
# -------------------------

def critic_node(state: GraphState) -> GraphState:
    student = state["student_response"]
    critic = state["critic_state"]

    progress_signal = "improving" if "fixed" in student.lower() else "stalled"
    diagnosis = (
        "You’re close; sign error in step 2." if "sign" in student.lower() else "Approach seems off."
    )

    needs_planner = (critic.get("turns_seen", 0) >= 3 and progress_signal != "improving")

    if needs_planner:
        state["feedback_or_plan"] = {
            "type": "request_plan",
            "progress": progress_signal,
            "diagnosis": diagnosis,
        }
    else:
        state["feedback_or_plan"] = {
            "type": "feedback",
            "fresh_start": critic.get("turns_seen", 0) == 0,
            "diagnosis": diagnosis,
            "misconceptions": ["sign_handling"] if "sign" in student.lower() else [],
            "next_hint": "Check the sign when distributing the negative.",
            "confidence": 0.62,
            "signals": {"progress": progress_signal, "attempt_quality": "medium"},
        }

    state["critic_state"] = update_switching_logic(critic, progress_signal)
    return state


def planner_node(state: GraphState) -> GraphState:
    diagnosis = state["feedback_or_plan"].get("diagnosis", "") if state.get("feedback_or_plan") else ""
    plan = {
        "type": "plan",
        "strategy": "worked_example" if "sign" in diagnosis else "socratic",
        "rationale": "Target common sign misconception with a short worked example.",
        "candidate_questions": [
            "What is -1 × (x - 3)?",
            "Where does the negative distribute?",
        ],
        "hints": ["Distribute the negative before combining like terms."],
        "examples": [{"problem": "-(x-3) + 2x", "solution_sketch": "-x + 3 + 2x → x + 3"}],
        "difficulty_delta": -1,
        "scaffolding_strength": "medium",
        "timeout_to_review": 2,
    }
    state["feedback_or_plan"] = plan
    return state


def s1_node(state: GraphState) -> GraphState:
    fop = state["feedback_or_plan"] or {}
    if fop.get("type") == "feedback":
        hint = fop.get("next_hint", "Focus on one step at a time.")
    else:
        hints = fop.get("hints", ["Focus on one step at a time."])
        hint = hints[0] if hints else "Focus on one step at a time."
    tutor_utterance = f"Hint: {hint} Try simplifying the expression step by step."
    state["episodic"]["turns"].append({"role": "tutor", "content": tutor_utterance})
    return state


def s2_node(state: GraphState) -> GraphState:
    fop = state["feedback_or_plan"] or {}
    examples = fop.get("examples", []) if fop.get("type") == "plan" else []
    if examples:
        ex = examples[0]
        tutor_utterance = (
            f"Let's walk a worked example: {ex['problem']} → {ex['solution_sketch']}. Now apply this pattern."
        )
    else:
        tutor_utterance = (
            "Let’s reason carefully. Identify operations, then apply in order. Where could sign flip?"
        )
    state["episodic"]["turns"].append({"role": "tutor", "content": tutor_utterance})
    return state


def route_from_critic(state: GraphState) -> str:
    critic = state["critic_state"]
    fop = state.get("feedback_or_plan")
    if fop and fop.get("type") == "request_plan":
        return "planner"
    return "s2" if critic.get("current_mode", "S1") == "S2" else "s1"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("critic", critic_node)
    graph.add_node("planner", planner_node)
    graph.add_node("s1", s1_node)
    graph.add_node("s2", s2_node)

    graph.set_entry_point("critic")
    graph.add_conditional_edges(
        "critic", route_from_critic, {"planner": "planner", "s1": "s1", "s2": "s2"}
    )
    graph.add_edge("planner", "s1")
    graph.add_edge("planner", "s2")
    graph.add_edge("s1", END)
    graph.add_edge("s2", END)

    return graph.compile()


def initial_state(student_message: str) -> GraphState:
    return {
        "student_response": student_message,
        "episodic": {"turns": [{"role": "student", "content": student_message}]},
        "critic_state": {
            "turns_seen": 0,
            "current_mode": "S1",
            "consecutive_failures": 0,
            "recent_misconceptions": [],
            "last_strategy": None,
            "last_confidence": 0.0,
        },
        "feedback_or_plan": None,
    }


def run_one_turn(student_message: str) -> Dict[str, Any]:
    graph = build_graph()
    state = initial_state(student_message)
    result = graph.invoke(state)
    return result


if __name__ == "__main__":
    # Simple CLI demo
    import sys

    msg = "".join(sys.argv[1:]) or "I think I messed up the sign when expanding."
    final_state = run_one_turn(msg)
    last_tutor_msg = final_state["episodic"]["turns"][-1]["content"]
    print(last_tutor_msg)


