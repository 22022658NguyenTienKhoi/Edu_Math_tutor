import os
import json
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data_models import (
    TutoringSessionState, ErrorAnalysis, SynthesizerReport,
)
from tools import query_knowledge_base

# --- Cấu hình môi trường ---
import dotenv
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

# Sử dụng một model mạnh mẽ hơn cho các tác vụ suy luận phức tạp
# Sử dụng một model nhanh hơn cho các tác vụ đơn giản hoặc sáng tạo
MAX_REGENERATION_ATTEMPTS = 2
# --- Cấu hình môi trường ---
import dotenv
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

# Model configuration (parameterized for flexibility)
DEFAULT_REASONING_MODEL = "models/gemma-3-27b-it"
#"models/gemini-flash-lite-latest"
DEFAULT_FAST_MODEL = "models/gemma-3-27b-it"
MAX_REGENERATION_ATTEMPTS = 2

import re, json
from langchain.schema import BaseOutputParser

class ReasoningJsonParser(BaseOutputParser):

    def __init__(self, pydantic_object: any):
        super().__init__()
        self._pydantic_object = pydantic_object

    def parse(self, text: str):
        # Try to extract the first JSON block
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None

            json_str = match.group(0)

            # Try to parse the JSON
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None

            # Try to create the pydantic object
            return self._pydantic_object(**data)
        except Exception as e:
            return None

class ErrorAgent:
    def __init__(self, reasoning_model=DEFAULT_REASONING_MODEL, fast_model=DEFAULT_FAST_MODEL):
        # LLM cho các tác vụ suy luận, phân tích sâu
        self.llm_reasoning = ChatGoogleGenerativeAI(model=reasoning_model, temperature=0.5, top_k = 50)
        # LLM cho các tác vụ sáng tạo, sinh câu hỏi
        self.llm_creative = ChatGoogleGenerativeAI(model=fast_model, temperature=0.5, top_k = 50)

    def _create_chain(self, prompt_template: str, pydantic_object: any, llm, output_type="json"):
        prompt = PromptTemplate.from_template(prompt_template)

        if output_type == "json":
            base_parser = ReasoningJsonParser(pydantic_object=pydantic_object)
            return prompt | llm | base_parser

        elif output_type == "str":
            return prompt | llm | StrOutputParser()

    # --- NODE: Các Agent Phát hiện lỗi ---
    
    def run_detectors(self, state: TutoringSessionState) -> dict:
        """Chạy tất cả các detector song song (giả lập)."""
        print("--- NODE: Running All Error Detectors ---")
        student_solution = state.initial_student_solution
        problem = state.problem_statement

        # Tạo các chain cho từng detector
        calc_chain = self._create_chain(
            '''Phân tích các bước giải sau đây và chỉ ra lỗi tính toán.
            Các lỗi có thể bao gồm:
                Calculation Error (CAL): tính toán sai
                Operation Error (OP): nhầm lẫn phép toán (+, -, *, /).
                Counting Error (CO): lỗi sai đếm số lượng, chỉ số.
            Pay attention to the detailed calculations.
            Bài toán: '{problem}'. Bài giải: '{solution}'.
            Thinking step by step.
            Trả về một đối tượng JSON CHỈ CHỨA các trường:
            contains_error: bool = Field(description="True nếu phát hiện có lỗi thuộc loại này.")
            explanation: str = Field(description="Giải thích ngắn gọn về lỗi được tìm thấy.")
            sub_error: str = Field(description="Loại lỗi phân loại chi tiết.")''',
            ErrorAnalysis, self.llm_reasoning)
        concept_chain = self._create_chain(
            '''Phân tích xem học sinh có áp dụng sai hoặc quên khái niệm toán học nào không. Kiến thức tham khảo: {knowledge}. Bài toán: '{problem}'. Bài giải: '{solution}'. \
            Các lỗi có thể bao gồm:
                Formula Confusion Error (FC): lỗi nhầm công thức.
                Knowledge Error (KNOW): lỗi thiếu/nhầm kiến thức.
                Context Value Error (CV): lỗi sai thuộc tính giá trị.
            Thinking step by step
            Pay attention to the detailed steps.
            Trả về một đối tượng JSON CHỈ CHỨA các trường:
            contains_error: bool = Field(description="True nếu phát hiện có lỗi thuộc loại này.")
            explanation: str = Field(description="Giải thích ngắn gọn về lỗi được tìm thấy.")
            sub_error: str = Field(description="Loại lỗi phân loại chi tiết.")''',
            ErrorAnalysis, self.llm_reasoning)
        logic_chain = self._create_chain(
            '''Phân tích mạch tư duy của bài giải. Các bước có logic không? Có bước nào bị bỏ qua hay suy luận vô lý không? Bài toán: '{problem}'. Bài giải: '{solution}'." \
            Các lỗi có thể bao gồm:
                Reasoning Error (REAS): lỗi sai lý luận.
                Missing Step (MS): lỗi thiếu bước trung gian.
                Contradictory Step (CS): lỗi bước mâu thuẫn.
                Hallucination (HA): lỗi thêm thông tin bịa/không liên quan.
            Thinking step by step.
            Pay attention to the detailed steps.
            Trả về một đối tượng JSON CHỈ CHỨA các trường:
            contains_error: bool = Field(description="True nếu phát hiện có lỗi thuộc loại này.")
            explanation: str = Field(description="Giải thích ngắn gọn về lỗi được tìm thấy.")
            sub_error: str = Field(description="Loại lỗi phân loại chi tiết.")''',
            ErrorAnalysis, self.llm_reasoning)
        comprehension_chain = self._create_chain(
            '''Phân tích xem bài giải có đi lạc đề hay hiểu sai yêu cầu của bài toán không. Bài toán: '{problem}'. Bài giải: '{solution}'." \
            Thinking step by step.
                Misinterpretation of the Question (MIS): lỗi hiểu sai yêu cầu đề.
            Trả về một đối tượng JSON CHỈ CHỨA các trường:
            contains_error: bool = Field(description="True nếu phát hiện có lỗi thuộc loại này.")
            explanation: str = Field(description="Giải thích ngắn gọn về lỗi được tìm thấy.")
            sub_error: str = Field(description="Loại lỗi phân loại chi tiết.")''',
            ErrorAnalysis, self.llm_reasoning)

        # Giả lập truy vấn kiến thức cho concept_chain
        knowledge = query_knowledge_base(problem)

        # Invoke các chain (trong thực tế có thể dùng LangChain parallel execution)
        reports = {
            "calculation": calc_chain.invoke({"problem": problem, "solution": student_solution}),
            "conceptual": concept_chain.invoke({"problem": problem, "solution": student_solution, "knowledge": knowledge}),
            "logic": logic_chain.invoke({"problem": problem, "solution": student_solution}),
            "comprehension": comprehension_chain.invoke({"problem": problem, "solution": student_solution})
        }
        
        #print(f"Detector Reports: {reports}")
        return {"latest_detector_reports": reports}

    # --- NODE: Agent Tổng hợp ---

    def synthesizer(self, state: TutoringSessionState) -> dict:
        """Tổng hợp kết quả từ các detector để đưa ra một báo cáo toàn diện."""
        #print("--- NODE: Synthesizer ---")
        chain = self._create_chain(
            """Bạn là Agent Tổng hợp. Nhiệm vụ của bạn là tạo một báo cáo tổng thể từ các detector. 
               báo cáo TẬP TRUNG vào phân tích lỗi sai của học sinh, KHÔNG tập trung vào phản hồi cho học sinh.

               Input:
                - Detector Reports: {detector_reports}

               Yêu cầu:
                1. Từ các lỗi trong detector_reports, hãy xác định các loại lỗi chính cần được ưu tiên khắc phục.
                Dưới đâu là danh sách tất cả các lỗi:
                    CAL: Lỗi tính toán số học (Arithmetic miscalculation)
                    OP: Sử dụng toán tử không đúng (Operator misuse)
                    CO: Lỗi đếm, sai số lượng hoặc chỉ số (Counting/index mistake)
                    FC: Nhầm lẫn hoặc áp dụng sai công thức (Formula confusion)
                    KNOW: Nhớ hoặc hiểu sai kiến thức (Knowledge misunderstanding)
                    CV: Gán sai thuộc tính giá trị (Wrong value attribution)
                    REAS: Lý luận sai, suy luận không đúng (Faulty reasoning)
                    MS: Thiếu bước trung gian quan trọng (Missing step)
                    CS: Các bước mâu thuẫn lẫn nhau (Contradiction in steps)
                    HA: Thêm thông tin bịa/không liên quan (Hallucinated info)
                    MIS: Hiểu sai yêu cầu đề bài (Misunderstood the question)
                Thinking step by step.
                Output:
                Trả về một đối tượng JSON với schema:
                "detailed_analysis": str,             // Phân tích chi tiết các lỗi trong bài làm.
                "primary_error_type": trả về danh sách các lỗi trong bài làm. Các lỗi phải thuộc 'CAL', 'CO', 'OP', 'FC', 'KNOW', 'CV', 'REAS', 'MS', 'CS', 'HA', 'MIS'
                """,
            SynthesizerReport, self.llm_creative
        )
        report = chain.invoke({"detector_reports": state.latest_detector_reports, "problem": state.problem_statement, 'solution': state.initial_student_solution})
        print(f"Synthesizer Report: {report}")
        return {"latest_synthesizer_report": report,
                "synthesizer_history": state.synthesizer_history + [report]}
    
def get_error_agent():
    workflow = ErrorAgent()
    
    builder = StateGraph(TutoringSessionState)

    # Thêm các node vào graph
    builder.add_node("run_detectors", workflow.run_detectors)
    builder.add_node("synthesizer", workflow.synthesizer)

    # Thiết lập điểm bắt đầu
    builder.set_entry_point("run_detectors")

    # Kết nối các node
    builder.add_edge("run_detectors", "synthesizer")
    
    builder.add_edge("synthesizer", END)

    #memory = MemorySaver()checkpointer=memory

    return builder.compile()