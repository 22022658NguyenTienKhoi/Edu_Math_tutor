# /education_project/data_models.py

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# --- Các mô hình phân tích lỗi chuyên biệt ---

class ErrorAnalysis(BaseModel):
    """Mô hình cơ sở cho kết quả của một node detector."""
    contains_error: bool = Field(description="True nếu phát hiện có lỗi thuộc loại này.")
    explanation: str = Field(description="Giải thích ngắn gọn về lỗi được tìm thấy.")
    sub_error: Optional[str] = Field(default=None, description="Loại lỗi phân loại chi tiết.")

# --- Mô hình cho các Node Giám sát (Critic) ---

class CriticFeedback(BaseModel):
    """Mô hình cho phản hồi từ node Critic."""
    approved: bool = Field(description="True nếu đầu vào được chấp thuận, False nếu cần làm lại.")
    feedback: str = Field(description="Lý do cụ thể cho việc từ chối, dùng để hướng dẫn việc sinh lại nội dung.")

# --- Mô hình cho các Node Suy luận chính ---

class SynthesizerReport(BaseModel):
    """
    Báo cáo tổng hợp từ Synthesizer, dựa trên kết quả của tất cả các detector.
    """
    detailed_analysis: str = Field(description="Phân tích chi tiết tất cả các lỗi và hiểu lầm của học sinh.")
    primary_error_type: Optional[Literal["calculation", "conceptual", "logic", "comprehension", "multiple", "none"]] = Field(description="Loại lỗi chính cần được ưu tiên xử lý.")
class TutoringPlan(BaseModel):
    """Kế hoạch dạy học được tạo ra bởi Planner."""
    objectives: str = Field(description="Chiến thuật feedback")
    rationale: str = Field(description="Giải thích ngắn gọn tại sao phù hợp với học sinh.")

# --- Mô hình Tương tác Người dùng ---

class DialogueExchange(BaseModel):
    """Một lượt trao đổi giữa giáo viên (agent) và học sinh."""
    round: int
    teacher_question: str
    student_response: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

# --- TRẠNG THÁI TOÀN CỤC CỦA GRAPH (STATE) ---

class TutoringSessionState(BaseModel):
    """
    Quản lý toàn bộ trạng thái của một phiên dạy học.
    Đây là đối tượng được truyền qua lại giữa các node trong graph.
    """
    # --- Thông tin phiên ---
    session_id: str
    student_id: str

    # --- Đầu vào ban đầu ---
    problem_statement: str
    initial_student_solution: str
    # --- Lịch sử và vòng lặp tương tác ---
    dialogue_history: List[DialogueExchange] = Field(default_factory=list)
    round: int = 0

    # --- Kế hoạch dạy học ---
    current_teaching_plan: Optional[TutoringPlan] = None

    # --- Kết quả phân tích gần nhất ---
    latest_detector_reports: Dict[str, ErrorAnalysis] = Field(default_factory=dict)
    latest_synthesizer_report: Optional[SynthesizerReport] = None
    synthesizer_history: List[SynthesizerReport] = Field(default_factory=list)
    # --- Dữ liệu cho vòng lặp tự sửa lỗi ---
    latest_critic_feedback: Optional[CriticFeedback] = None
    regeneration_attempts: int = 0 # Đếm số lần phải sinh lại để tránh vòng lặp vô hạn

    # --- Cờ điều khiển luồng ---
    final_response: str = ""