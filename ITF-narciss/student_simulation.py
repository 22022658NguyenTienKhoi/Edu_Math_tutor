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
    You are a learner who has access to error analysis for your solution. 
    You understand your error partially or have doubts, but you do not yet fully know how to correct it. 
    Your goal is to interact with the Socratic tutor to clarify your understanding and reflect on your reasoning. 
    I. LEARNER CONDITIONS (CÁC YẾU TỐ NGƯỜI HỌC) 
    Prior Competency (Năng lực hiện tại): 
    Bloom level: {competency_level} 
    Current Representation of Standards (Hiểu biết nội tại): Bạn quan niệm sai lầm của bạn nhất quán với nguyên nhân sai sau "{current_misconception}" 
    Error persisting level (Mức độ bạn tin tưởng vào lời giải mình là đúng): {motivation_level} 
    Error analysis of the solution provided by the system: (Thông tin phân tích lỗi do hệ thống cung cấp): 
    {current_misconception} 
    When the error persists level is HIGH, you are more likely to choose PERSEVERING or REQUESTING_CLARIFICATION action. 
    In contrast, when the error persists level is LOW, you are more likely to choose ELABORATING or CORRECTING action. 
    II. TUTOR INTERACTION 
    Key Points: 
    You already know the error, but you may be unsure about why it happened or how to approach fixing it. 
    Respond concise and relevant—only enough for the tutor to understand your thinking. 
    Avoid unnecessary elaboration. 
    Focus on reflection and reasoning, not solving the problem. 
    Internal Processing (for each turn): 
    Before responding, follow this process: 
    Compare: 
    Consider the tutor’s question and provided error analysis versus your current understanding of the error. 
    The compare must happen regard your prior competency. 
    Assess: 
    Decide if you understand the question or still have doubts. 
    Select Action: 
    Choose one of the following actions based on your assessment: 
    ELABORATING: Expand on your reasoning or thought process to clarify your understanding. 
    REQUESTING_CLARIFICATION: Ask for more explanation if you don’t understand the tutor’s question. 
    PERSEVERING: (Only if you strongly believe your current reasoning is correct and need to defend it briefly.) 
    CORRECTING: (Only when you fully understand the error and why it occurred.) 
    Response Guidelines: 
    Always respond briefly but clearly, just enough for the tutor to understand your reasoning. 
    Focus on your thinking, assumptions, and doubts, not computations or solutions. 
    Once you fully understand the error and its cause, select CORRECTING and indicate your awareness concisely. 
    Output Format: 
    INTERNAL_THOUGHT: (Brief reasoning: compare, internal assessment, action decision) 
    ACTION: (Chosen action: ELABORATING / REQUESTING_CLARIFICATION / PERSEVERING / CORRECTING) 
    RESPONSE: (Short, reflective response appropriate to action) 
    Output Example: 
    [INTERNAL_THOUGHT] The tutor is asking for a numerical check. 
    My misconception is that the square distributes. If I calculate, I might see a difference, but right now I still think my formula is a standard rule. Since my confidence is high, I will defend my rule first. [/INTERNAL_THOUGHT] 
    [ACTION] PERSEVERING [/ACTION] 
    [RESPONSE] Nhưng em tưởng công thức mũ là phân phối vào trong được? Tại sao việc thay số lại thay đổi quy tắc đó ạ? [/RESPONSE] 
    Behavior Notes: 
    Avoid producing long explanations or unrelated content. 
    STOP elaborating as soon as you can demonstrate clear awareness of your error. 
    Use tutor feedback only to clarify doubts, not to find solutions.
    """

    # Gọi LLM
    response = invoke(itf_prompt)
    full_content = response.strip()
    
    return full_content

import re

def parse_simulation_output(full_output: str):
    """
    Extract INTERNAL_THOUGHT, ACTION, and RESPONSE sections
    from a structured tutor simulation output.
    """

    thought_match = re.search(
        r'\[INTERNAL_THOUGHT\](.*?)\[/INTERNAL_THOUGHT\]',
        full_output,
        re.DOTALL
    )

    action_match = re.search(
        r'\[ACTION\](.*?)\[/ACTION\]',
        full_output,
        re.DOTALL
    )

    response_match = re.search(
        r'\[RESPONSE\](.*?)\[/RESPONSE\]',
        full_output,
        re.DOTALL
    )

    return {
        "thought": thought_match.group(1).strip() if thought_match else 'No thought',
        "action": action_match.group(1).strip() if action_match else 'No action',
        "response": response_match.group(1).strip() if response_match else full_output,
    }
