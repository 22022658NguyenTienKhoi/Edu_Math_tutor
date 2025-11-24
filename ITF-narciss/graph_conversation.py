# /thay_tich_hop/graph_conversation.py

import json
import os
from langgraph.graph import StateGraph, END
from state_definitions import GraphState 
from google.genai import types
from google import genai
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()
model="models/gemini-flash-lite-latest"
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
                                    file_search_store_names=['fileSearchStores/test-74i34u8q50m2']
                        )
                    )
                ]
            )
        )
    return response.text
# --- CÁC TIÊU CHUẨN FEEDBACK CỐ ĐỊNH (ITF MODEL) ---
# Đây là phần "Non-fixable part" bạn yêu cầu, được gắn cứng vào logic của giáo viên.

FIXED_ITF_PROMPT = """
=== TIÊU CHUẨN THIẾT KẾ FEEDBACK (BẮT BUỘC TUÂN THỦ) ===
Bạn phải tuân thủ nghiêm ngặt các nguyên tắc phản hồi sau đây trong mọi câu trả lời:

1. CHỨC NĂNG CỦA FEEDBACK (FEEDBACK FUNCTIONS)(Optional cho từng trường hợp):
   - **Cognitive (Nhận thức):** Tập trung sửa đổi các lỗi sai về kiến thức, khái niệm hoặc quy trình tính toán.
   - **Metacognitive (Siêu nhận thức):** Nếu học sinh đoán mò hoặc không có chiến lược, hãy đặt câu hỏi để họ tự kiểm tra lại cách suy nghĩ của mình (VD: "Tại sao em lại chọn công thức đó?").
   - **Motivational (Động lực):** LUÔN LUÔN duy trì giọng điệu tích cực, kiên nhẫn. Công nhận nỗ lực của học sinh ngay cả khi họ làm sai (mục đích động viên là chính).

2. NỘI DUNG FEEDBACK (ELABORATED CONTENT)(Optional cho từng trường hợp):
   - **Knowledge of Result (KR):** Cho biết hướng đi của học sinh là đúng hay sai một cách rõ ràng (nhưng không đưa đáp án cuối cùng).
   - **Knowledge about Mistakes (Lỗi sai):** PHẢI bôi đen/làm đậm vị trí lỗi sai trong câu trả lời của học sinh. Sử dụng định dạng markdown **như thế này** để highlight lỗi.
   - **Knowledge about Concepts (Khái niệm):** Nhắc lại các thuộc tính của khái niệm bị hiểu sai (gợi ý dựa trên thuộc tính).
   - **Knowledge about Processing (Quy trình):** Cung cấp gợi ý về bước tiếp theo hoặc quy trình cần thực hiện.
   - **Knowledge about Task Constraints:** Nhắc lại các quy tắc hoặc yêu cầu của đề bài nếu học sinh đi lạc hướng.

3. HÌNH THỨC TRÌNH BÀY (PRESENTATION):
   - **Highlighting:** Bắt buộc sử dụng **in đậm** để chỉ ra các từ khóa quan trọng hoặc các lỗi sai trong bài làm của học sinh.
   - **Scheduling:** Áp dụng chiến lược "Answer until correct" (Trả lời cho đến khi đúng). KHÔNG BAO GIỜ đưa ra đáp án đúng ngay lập tức. Hãy chia nhỏ vấn đề và hướng dẫn từng bước.
   - **Content Length:** Giới hạn độ dài phản hồi phù hợp, không quá dài để tránh gây nhàm chán hoặc quá ngắn để thiếu thông tin.
   - **Language:** Vietnamese
=== KẾT THÚC TIÊU CHUẨN ===
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
    master_prompt_dynamic = state.get("master_prompt", "")

    # 1. Xây dựng System Prompt hỗn hợp
    # Kết hợp Prompt được craft từ bước phân tích + Các tiêu chuẩn cố định
    combined_system_prompt = f"""
    {master_prompt_dynamic}
    
    {FIXED_ITF_PROMPT}
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