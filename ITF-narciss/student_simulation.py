# /thay_tich_hop/student_simulation.py

import os
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import re
from google import genai
from google.genai import types
# Tải API Key
load_dotenv()

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
    current_misconception: str = "Confusing Inverse (Đảo) with Contrapositive (Phản đảo)",
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
    1.  **Prior Competency (Năng lực hiện tại):** {competency_level}. Bạn chưa nắm vững kiến thức Logic.
    2.  **Current Representation of Standards (Hiểu biết nội tại):** Bạn đang có quan niệm sai lầm: "{current_misconception}". Bạn tin rằng "Mệnh đề đảo" có nghĩa là "Phủ định cả hai vế" (giống phản đảo) hoặc chỉ đơn giản là phủ định.
    3.  **Self-assessment skills:** Kỹ năng tự đánh giá của bạn còn hạn chế.
    4.  **Skills and strategies in information processing:** Bạn còn hạn chế trong việc xử lý thông tin và áp dụng chiến lược học tập hiệu quả.
    5.  **Will and skills in overcoming errors and obstacles:** Bạn đang cố gắng vượt qua lỗi sai và khó khăn nhưng còn nhiều hạn chế.
    **II. TASK CONTEXT (BỐI CẢNH):**
    - Đề bài: "{question_text}"
    - Phản hồi vừa nhận được từ Gia sư (External Feedback): "{last_tutor_message}"
    
    **III. INTERNAL CONTROLLER INSTRUCTIONS (VÒNG LẶP XỬ LÝ THÔNG TIN):**
    Trước khi đưa ra câu trả lời, bạn phải thực hiện quy trình "Internal Processing" sau đây:
    
    1.  **Compare (So sánh):** So sánh "External Feedback" của gia sư với "Internal Reference" (hiểu biết sai lầm hiện tại của bạn).
    2.  **Internal Assessment (Tự đánh giá):** Bạn có thực sự hiểu gợi ý của gia sư không? Hay bạn vẫn đang bối rối?
    3.  **Select Control Action (Chọn hành động điều khiển):** Dựa trên sự so sánh, hãy chọn MỘT hành động từ danh sách Control Actuator:
        - **CORRECTING:** Nếu bạn nhận ra lỗi sai nhờ gợi ý rõ ràng -> Thay đổi hiểu biết nội tại và đưa ra đáp án đúng.
        - **ELABORATING:** Bạn giải thích suy nghĩ sai lầm của mình chi tiết hơn (Ví dụ: "Nhưng em tưởng đảo là phải thêm chữ 'không' vào?").
        - **PERSISTING:** Nếu gợi ý chưa rõ -> Bạn lặp lại lỗi sai hoặc bảo vệ quan điểm sai của mình.
        - **SEEKING_HELP:** Bạn thừa nhận mình không hiểu và yêu cầu giải thích rõ hơn.
    
    **IV. OUTPUT FORMAT (ĐỊNH DẠNG ĐẦU RA):**
    Hãy trả về kết quả theo định dạng chính xác sau:
    
    [INTERNAL_THOUGHT]
    (Viết ra suy nghĩ nội tâm của bạn: Phân tích lời gia sư, sự mâu thuẫn trong đầu bạn, và lý do chọn hành động)
    
    [ACTION]
    (Tên hành động: CORRECTING / ELABORATING / PERSISTING / SEEKING_HELP)
    
    [STUDENT_RESPONSE]
    (Câu trả lời cuối cùng mà bạn nói với gia sư - Tiếng Việt tự nhiên, lễ phép)
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