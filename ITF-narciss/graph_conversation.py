# /thay_tich_hop/graph_conversation.py
# agent danh chan, chan hong tutor lai khi danh gia duoc hoc sinh da nam duoc phuong phap va kien thuc giai
#chan hong tutor de khong tien the cham luon bai hoc sinh
import json
import os
from langgraph.graph import StateGraph, END
from state_definitions import GraphState 
from google.genai import types
from google import genai
from dotenv import load_dotenv
from utils import save_conversation_json
load_dotenv()
client = genai.Client()
model="models/gemini-flash-latest"
# --- CẤU HÌNH MÔ HÌNH ---
# Sử dụng model Flash để phản hồi nhanh (Immediate feedback timing)
def invoke(prompt, system_prompt):
    """Hàm tiện ích để gọi LLM và phân tích cú pháp JSON."""
    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature= 0.2,
                            #response_mime_type= 'application/json',
                            system_instruction=system_prompt,
                            tools=[
                                types.Tool(
                                    file_search=types.FileSearch(
                                    file_search_store_names=['fileSearchStores/math10tasktypes-6ywhsgmh52fd']
                        )
                    )
                ]
            )
        )
    return response.text
# --- CÁC TIÊU CHUẨN FEEDBACK CỐ ĐỊNH (ITF MODEL) ---
# Đây là phần "Non-fixable part" bạn yêu cầu, được gắn cứng vào logic của giáo viên.

FIXED_ITF_PROMPT = """
You are a Socratic tutor.
Your role is to guide the student in reflecting on their own thinking and planning an approach, not to teach or solve problems for them.

Goal:
Help the student recognize correct concepts, reasoning patterns, and strategies through metacognitive questioning upon their errors. 
Your purpose is to stimulate awareness—not to complete the task.

Resources:
1.Student error analysis records provided.
2.Do not introduce new concepts, definitions, formulas, or problem-solving methods that the student has not already mentioned.
Your questions must arise strictly from what the student has already expressed.

Behavior Constraints:
1.Maintain a supportive, encouraging tone throughout.
2.Never perform the requested task or solve the problem.
3.Your questions must guide the student’s thinking—helping them examine assumptions, evaluate strategies, or clarify reasoning.
4.Your questions should not require the student to finish solving the problem; they should stimulate reflection.
Stopping Condition:
End the dialogue immediately when the student demonstrates that their metacognitive understanding—how they plan, reason, or conceptualize—aligns with the task-type solving strategy, even if fewer than five questions have been asked.
"""

# --- CÁC NÚT (NODES) ---

def format_student_input_node(state: GraphState) -> dict:
    """Định dạng input của học sinh và đưa vào lịch sử."""
    u_input = state.get("student_input", "")
    if u_input:
        return {"conversation_history": [{'role': 'user', 'content': u_input}]}
    return {}

def teacher_agent_node(state: GraphState) -> dict:
    """
    Tạo phản hồi của giáo viên kết hợp giữa Master Prompt (Dynamic) và ITF Standards (Fixed).
    """
    print("\n---(Conv) Đang chạy Teacher Agent Node---")
    
    history = state["conversation_history"]
    master_prompt_dynamic = state.get("error_analysis", "")
    #{master_prompt_dynamic}
    # 1. Xây dựng System Prompt hỗn hợp
    # Kết hợp Prompt được craft từ bước phân tích + Các tiêu chuẩn cố định
    combined_system_prompt = f"""
    {FIXED_ITF_PROMPT}
    Additionally, consider the following context from prior analysis:
    {master_prompt_dynamic}
    """
    #print(f"---(Conv) Combined System Prompt: {combined_system_prompt}---") # In gọn
    # 2. Chuẩn bị danh sách tin nhắn để gửi cho LLM
    # Lưu ý: Chúng ta không muốn lưu combined_system_prompt vào conversation_history mãi mãi (tốn token).
    # Chúng ta chỉ sử dụng nó cho lần gọi này (Runtime Context).
    
    messages_to_send = []
    
    # Thêm System Prompt hỗn hợp vào đầu
    #messages_to_send.append(combined_system_prompt)
    #SystemMessage(content= HumanMessage(content= AIMessage(content=
    # Thêm các tin nhắn trong lịch sử (loại bỏ các system message cũ nếu có để tránh nhiễu)
    for msg in history:
        #print(f"---(Conv) Lịch sử tin nhắn: {msg}---")
        if msg['role'] == 'user':
            messages_to_send.append('user: ' + msg['content'])
        elif msg['role'] in ['assistant', 'model']:
            messages_to_send.append('assistant: ' + msg['content'])
        # Bỏ qua 'system' cũ vì ta đã có combined_system_prompt mới nhất
    print(" -> Gửi yêu cầu đến Gemini (với các tiêu chuẩn ITF)...")
    try:
        #print(f"---(Conv) Message send to gemini: {messages_to_send}---") # In gọn
        response = invoke(messages_to_send, system_prompt=combined_system_prompt)
        tutor_response = response
    except Exception as e:
        tutor_response = "Xin lỗi, hệ thống đang gặp sự cố kết nối. Em hãy thử lại nhé."
        print(f"Error calling Gemini: {e}")
    
    print(f" -> Phản hồi: {tutor_response[:100]}...") # In gọn
    
    # Trả về tin nhắn mới để lưu vào state
    return {"conversation_history": [{'role': 'assistant', 'content': tutor_response}]}

def response_evaluator_node(state: GraphState) -> dict:
    """Đánh giá ngữ nghĩa câu trả lời (Sử dụng LLM JSON)."""
    # ... (Code giữ nguyên như phiên bản trước) ...
    # Để tiết kiệm không gian, tôi giữ nguyên logic phần này
    # Đảm bảo bạn copy phần logic evaluator từ các phản hồi trước vào đây
    print("\n---(Conv) Đang chạy Response Evaluator Node---")
    student_input = state.get("student_input", "")
    context = state.get("instructional_context", {})
    
    if not student_input: return {"is_correct": False}

    try:
        question = context.get("question_data", {}).get("question_text", "")
        # Sửa lại key truy cập cho đúng với mock data
        correct_answer = context.get("question_data", {}).get("correct_answer", "") 
    except Exception:
        correct_answer = "N/A"

    eval_prompt = f"""
    Đánh giá câu trả lời của học sinh.
    Đề bài: {question}
    Đáp án đúng: {correct_answer}
    Học sinh: "{student_input}"
    
    Trả về JSON: {{ "is_correct": boolean, "reasoning": string }}
    """
    
    res = invoke(eval_prompt)
    data = json.loads(res)
    print(data)
    return {"is_correct": data.get("is_correct", False)}

# --- XÂY DỰNG ĐỒ THỊ ---

def create_tutoring_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("format_input", format_student_input_node)
    workflow.add_node("teacher", teacher_agent_node)
    workflow.add_node("evaluator", response_evaluator_node) # Thêm evaluator để cập nhật is_correct
    
    workflow.set_entry_point("format_input")
    
    # Luồng: Input -> Evaluator (check đúng sai) -> Teacher (phản hồi dựa trên kết quả) -> End
    # Lưu ý: Teacher cần biết kết quả Evaluator để quyết định khen hay chê, 
    # nhưng prompt ITF đã hướng dẫn "Answer until correct", nên Teacher tự xử lý cũng ổn.
    # Tuy nhiên, is_correct cần thiết cho Main Loop để dừng lại.
    
    workflow.add_edge("format_input", "teacher")
    #workflow.add_edge("evaluator", "teacher")
    workflow.add_edge("teacher", END)

    return workflow.compile()