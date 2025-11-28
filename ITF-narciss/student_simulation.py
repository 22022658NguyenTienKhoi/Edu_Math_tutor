# /thay_tich_hop/student_simulation.py

import os
from dotenv import load_dotenv
import re
from google import genai
from google.genai import types
# Tải API Key
load_dotenv()
# 2 stage checking: sim stu have 4 action, 3 of them need tutor to respond again and one don't. If correcting is chosen, stop tutoring and eval solution.abs
#Metric: Number of actions each type taken.
# conversation till action is correcting, number of round, final correctness
# Khởi tạo model riêng cho học sinh
# Nên dùng temperature cao hơn một chút để tạo ra sự đa dạng trong cách trả lời
try:
    client = genai.Client()
    model="models/gemini-flash-lite-latest"
except Exception as e:
    print(f"Lỗi khởi tạo Student LLM: {e}")
    exit()

def invoke(prompt):
    """Hàm tiện ích để gọi LLM và phân tích cú pháp JSON."""
    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature= 0.5,
                        )
            )
    return response.text

def generate_simulated_student_response(
    conversation_history: list, 
    question_text: str, 
    current_misconception: str = "None",
    competency_level: str = "NOVICE (Người mới bắt đầu)",
    motivation_level: str = "MEDIUM (Muốn học nhưng dễ nản)"
) -> str:
    """
    Tạo câu trả lời mô phỏng dựa trên mô hình ITF (Interactive Tutoring Feedback).
    """
    
    # Lấy tin nhắn cuối cùng của gia sư
    last_tutor_message = conversation_history[-1]['content'] if conversation_history else "Bắt đầu bài học."
    
    # Chuyển đổi lịch sử hội thoại thành văn bản
    dialogue_text = ""
    for msg in conversation_history:
        role = "Gia sư (External Source)" if msg['role'] in ['system', 'model'] else "Học sinh (Learner)"
        if msg['role'] == 'system': continue
        dialogue_text += f"{role}: {msg['content']}\n"

    # --- XÂY DỰNG PROMPT DỰA TRÊN ITF MODEL ---
    
    itf_prompt = f"""
    BẠN ĐANG MÔ PHỎNG MỘT QUÁ TRÌNH NHẬN THỨC CỦA HỌC SINH (LEARNER) DỰA TRÊN MÔ HÌNH ITF CỦA NARCISS.
    
    **I. LEARNER CONDITIONS (CÁC YẾU TỐ NGƯỜI HỌC):**
    1.  **Prior Competency (Năng lực hiện tại):** Bloom level: {competency_level}.
    2.  **Current Representation of Standards (Hiểu biết nội tại):** Bạn quan niệm sai lầm của bạn nhất quán với nguyên nhân sai sau "{current_misconception}".
    3.  **Will and skills in overcoming errors and obstacles:** {motivation_level}
    **II. TASK CONTEXT (BỐI CẢNH):**
    - Đề bài: "{question_text}"
    - Phản hồi vừa nhận được từ Gia sư (External Feedback): "{last_tutor_message}"
    - Thông tin về những gì gia sư có thể làm và không thể làm: 
        1.Bạn chỉ nhận được các câu hỏi gợi mở từ gia sư, không có câu trả lời trực tiếp hay giải thích nào.
        2.Gia sư không thể giúp bạn hoàn thành bài tập, mà chỉ giúp bạn suy nghĩ về cách tiếp cận.
        3.Gia sư sẽ không giúp bạn các yêu cầu nằm ngoài phạm vi lỗi sai và hiểu biết hiện tại của bạn.
    - Lịch sử hội thoại trước đó giữa bạn và gia sư:
    {dialogue_text}
    **III. INTERNAL CONTROLLER INSTRUCTIONS (VÒNG LẶP XỬ LÝ THÔNG TIN):**
    Trước khi đưa ra câu trả lời, bạn phải thực hiện quy trình "Internal Processing" sau đây:
    
    1.  **Compare (So sánh):** So sánh "External Feedback" của gia sư với "Internal Reference" (hiểu biết sai lầm hiện tại của bạn).
    2.  **Internal Assessment (Tự đánh giá):** Bạn có thực sự hiểu gợi ý của gia sư không? Hay bạn vẫn đang bối rối?
    3.  **Select Control Action (Chọn hành động điều khiển):** Dựa trên sự so sánh, hãy chọn MỘT hành động từ danh sách Control Actuator:
        - **CORRECTING:** Bạn nhận ra lỗi sai trong bài và kết thúc mô phỏng.
        - **ELABORATING:** Bạn đi theo gợi ý của gia sư để tìm nguyên nhân lỗi sai.
        
    **IV. OUTPUT FORMAT (ĐỊNH DẠNG ĐẦU RA):**
    Hãy trả về kết quả theo định dạng chính xác sau:
    
    [INTERNAL_THOUGHT]
    (Viết ra suy nghĩ nội tâm của bạn: Phân tích lời gia sư, sự mâu thuẫn trong đầu bạn, và lý do chọn hành động)
    
    [ACTION]
    (Tên hành động: CORRECTING / ELABORATING)
    
    [STUDENT_RESPONSE]
    (Câu trả lời dựa trên hành động)
    """

    # Gọi LLM
    response = invoke(itf_prompt)
    full_content = response.strip()
    
    return full_content

def parse_simulation_output(full_output: str):
    """Hàm tiện ích để tách phần suy nghĩ và phần trả lời"""
    thought_match = re.search(r'\[INTERNAL_THOUGHT\](.*?)(?=\[ACTION\])', full_output, re.DOTALL)
    action_match = re.search(r'\[ACTION\](.*?)(?=\[STUDENT_RESPONSE\])', full_output, re.DOTALL)
    response_match = re.search(r'\[STUDENT_RESPONSE\](.*)', full_output, re.DOTALL)
    
    return {
        "thought": thought_match.group(1).strip() if thought_match else "No thought generated",
        "action": action_match.group(1).strip() if action_match else "UNKNOWN",
        "response": response_match.group(1).strip() if response_match else full_output
    }