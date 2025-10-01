import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data_models import (
    TutoringSessionState, ErrorAnalysis, SynthesizerReport,
    TutoringPlan, CriticFeedback, DialogueExchange
)
from tools import get_student_profile
from error_detectors import get_error_agent
# --- Environment Setup ---
import dotenv
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

DEFAULT_REASONING_MODEL = "models/gemma-3-27b-it"
#models/gemini-2.5-flash
DEFAULT_FAST_MODEL = "models/gemma-3-27b-it"

class TutoringWorkflow:
    """
    Implements the AI tutor workflow with specialized agents and monitoring loop.
    Models can be customized for reasoning and creative tasks.
    """
    def __init__(self, reasoning_model=DEFAULT_REASONING_MODEL, fast_model=DEFAULT_FAST_MODEL):
        self.llm_reasoning = ChatGoogleGenerativeAI(model=reasoning_model, temperature=0.2)
        self.llm_creative = ChatGoogleGenerativeAI(model=fast_model, temperature=0.7)

    def _create_chain(self, prompt_template: str, pydantic_object: any, llm, output_type="json"):
        """Hàm trợ giúp tạo một chain hoàn chỉnh."""
        prompt = PromptTemplate.from_template(prompt_template)
        if output_type == "json" :
            return prompt | llm | JsonOutputParser(pydantic_object=pydantic_object)
        elif output_type == "str":
            return prompt | llm | StrOutputParser()
        
    # --- NODE: Agent Lập kế hoạch ---

    def planner(self, state: TutoringSessionState) -> dict:
        """Create or update the teaching plan."""
        print("--- NODE: Planner ---")
        try:
            student_profile = get_student_profile(state.student_id)
        except Exception as e:
            print(f"Error getting student profile: {e}")
            student_profile = "N/A"

        prompt_template = """Bạn là Agent Lập kế hoạch Sư phạm. 
Dựa trên phân tích lỗi và phản hồi của học sinh, hãy đề xuất chiến thuật phản hồi để hỗ trợ học sinh theo hướng "scaffolding" (luôn giúp học sinh tiến bộ nhưng KHÔNG bao giờ cho đáp án trực tiếp).  
Bài toán: {problem}  
Các lỗi chính trong bài làm ban đầu của học sinh cần khắc phục: {initial_mistakes}  
Lịch sử hội thoại gần nhất: {history}  
Hồ sơ học sinh: {student_profile}  
Bạn chỉ được chọn **1 trong các chiến thuật feedback** sau:  
1. **Positive feedback**  
   - Nhấn mạnh phần học sinh đã làm đúng, giải thích ngắn gọn vì sao đúng.  
   - Giúp củng cố kiến thức đúng và tạo động lực.  
   - Không tiết lộ đáp án phần sai.  
2. **Knowledge about concepts**  
   - Cung cấp gợi ý và giải thích về các khái niệm, định nghĩa, công thức liên quan.  
   - Giúp học sinh bổ sung nền tảng kiến thức còn thiếu.  
   - Không áp dụng trực tiếp để ra đáp án.  
3. **Procedural feedback**  
   - Định hướng học sinh theo các bước giải quyết vấn đề.  
   - Gợi ý bước tiếp theo, quy tắc cần dùng.  
   - Không thực hiện tính toán thay học sinh.  
4. **Error-specific feedback**  
   - Chỉ ra rõ loại sai sót (sai dấu, nhầm công thức, nhầm đơn vị, …).  
   - Giúp học sinh tự sửa lỗi.  
   - Không đưa đáp án đúng.  
5. **Knowledge on metacognition**  
   - Khuyến khích học sinh tự nhìn lại quá trình suy nghĩ.  
   - Đặt câu hỏi phản tư: "Em có chắc bước này hợp lý không?"  
   - Không giải thích chi tiết hộ học sinh.  
Hãy trả về một đối tượng JSON với các trường:  
    objectives: str = Field(description="chiến thuật feedback và định nghĩa của nó lấy đúng theo danh sách trên.")
    rationale: str = Field(description="Giải thích ngắn gọn tại sao phù hợp với học sinh.")
Ví dụ:
    objectives: Procedural feedback-Định hướng học sinh theo các bước giải quyết vấn đề. Gợi ý bước tiếp theo, quy tắc cần dùng. Không thực hiện tính toán thay học sinh.  
    rationale: Học sinh đã hiểu khái niệm nhưng sai ở bước tính toán, nên cần hướng dẫn quy trình giải đúng hơn.
"""
        chain = self._create_chain(prompt_template, TutoringPlan, self.llm_creative)
        # Robust extraction of initial mistakes
        initial_mistakes = "N/A"
        if getattr(state, "synthesizer_history", None):
            first_report = state.synthesizer_history[0]
            initial_mistakes = getattr(first_report, "detailed_analysis", "N/A")
        try:
            plan = chain.invoke({
                "problem": state.problem_statement,
                "history": state.dialogue_history[-1] if state.dialogue_history else "N/A",
                "student_profile": student_profile,
                "initial_mistakes": initial_mistakes,
            })
        except Exception as e:
            print(f"Error invoking planner chain: {e}")
            plan = None
        print(f"Generated Plan: {plan}")
        return {
            "current_teaching_plan": plan,
            "latest_critic_feedback": None,
            "regeneration_attempts": 0
        }

    # --- NODE: Agent Sinh nội dung ---
    
    def content_generator(self, state: TutoringSessionState) -> dict:
        """Generate question/hint for student based on current plan objectives."""
        print("--- NODE: Content Generator ---")
        objectives = getattr(state.current_teaching_plan, "objectives", [])
        prompt_template = """Dựa trên mục tiêu dạy học sau đây, hãy tạo ra phản hồi nội dung bằng tiếng Việt.
Đề bài: {problem}
Mục tiêu: {objectives}
Lịch sử hội thoại: {history}
Lỗi trong bai làm ban đầu của học sinh mà cần khắc phục: {initial_solution}
"""
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm_reasoning
        initial_mistakes = "N/A"
        if getattr(state, "synthesizer_history", None):
            first_report = state.synthesizer_history[0]
            initial_mistakes = getattr(first_report, "detailed_analysis", "N/A")
        try:
            question = chain.invoke({
                "problem": state.problem_statement,
                "initial_solution": initial_mistakes,
                "objectives": objectives,
                "history": state.dialogue_history[-1] if state.dialogue_history else "N/A",
            }).content
        except Exception as e:
            print(f"Error invoking content generator chain: {e}")
            question = "(Không thể sinh câu hỏi)"
        # Simplified dialogue history update
        exchange = DialogueExchange(round=state.round, teacher_question=question)
        if not state.dialogue_history or getattr(state.dialogue_history[-1], "student_response", None):
            history = state.dialogue_history + [exchange]
        else:
            history = state.dialogue_history[:-1] + [exchange]
        return {"dialogue_history": history, "latest_critic_feedback": None}

# --- Xây dựng Graph ---

def get_tutoring_graph(reasoning_model=DEFAULT_REASONING_MODEL, fast_model=DEFAULT_FAST_MODEL):
    workflow = TutoringWorkflow(reasoning_model=reasoning_model, fast_model=fast_model)
    builder = StateGraph(TutoringSessionState)
    builder.add_node("planner", workflow.planner)
    builder.add_node("content_generator", workflow.content_generator)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "content_generator")
    builder.add_edge("content_generator", END)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# --- Chạy thử nghiệm ---
if __name__ == "__main__":
    app = get_tutoring_graph()
    error_agent = get_error_agent()
    config = {"configurable": {"thread_id": "session_001"}}
    import pandas as pd
    df = pd.read_csv("Student_error.csv")
    initial_state = {
        "session_id": "session_001",
        "student_id": "student_123",
        "problem_statement": df.iloc[0]["Problem"],
        "initial_student_solution": df.iloc[0]["Student_solution"],
        "round": 0
    }
    print("\n--- TURN 1: Initial Student Solution ---")
    final_state = error_agent.invoke(initial_state, config=config)
    current_state = final_state
    final_state_2 = app.invoke(current_state, config=config)
    teacher_question_2 = final_state_2['dialogue_history'][-1].teacher_question
    print("\n--- AGENT TO STUDENT ---")
    print(teacher_question_2)
    while True:
        user_input = input("Student: ")
        if user_input.lower() == "quit":
            break
        current_state = final_state_2
        current_state['dialogue_history'][-1].student_response = user_input
        current_state['round'] += 1
        final_state_2 = app.invoke(current_state, config=config)
        teacher_question_2 = final_state_2['dialogue_history'][-1].teacher_question
        print("\n--- AGENT TO STUDENT ---")
        print(teacher_question_2)