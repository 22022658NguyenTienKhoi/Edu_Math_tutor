# /thay_tich_hop/main.py

import time
import warnings
from error_detection_analysis import create_error_analysis_agent
from graph_conversation import create_tutoring_graph
from student_simulation import generate_simulated_student_response, parse_simulation_output
from utils import save_conversation_json
import pandas as pd
import random
# Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

if __name__ == "__main__":
    # 1. Khởi tạo các Agent
    print("--- Đang khởi tạo các tác nhân AI... ---")
    error_analysis_agent = create_error_analysis_agent()
    tutoring_graph = create_tutoring_graph()

    # 2. Dữ liệu ban đầu
    #question_text = "Viết phương trình tiếp tuyến của đường tròn (C): $(x+2)^2 + (y-3)^2 = 5$ tại điểm M(-1;1)."
    input_solution_df = pd.read_csv('final_dataset_with_topic.csv')
    rand = random.randint(0, len(input_solution_df)-1)
    print(f"Selected input index: {rand}")
    input_solution = input_solution_df.iloc[rand]
    question_text = input_solution['question']
    initial_solution = input_solution['wrong_solution']
    
    initial_state = {
        "student_id": "An (Simulated)",
        "question": question_text,
        "initial_student_solution": initial_solution,
        "student_input": initial_solution, 
        "round": 0,
        "conversation_history": [],
        "is_correct": False,
        "is_final": False,
        "instructional_context": {},
        "error_analysis": {},
        "learner_profile": {},
        "master_prompt": "",
    }

    # 3. Chạy Phân tích lỗi (Giai đoạn 1)
    print("="*50 + "\n🤖 BẮT ĐẦU MÔ PHỎNG VỚI HỌC SINH AI\n" + "="*50)
    print(f"📝 Đề bài: {input_solution['question']}")
    print(f"❌ Bài làm sai ban đầu: {initial_solution}\n")
    
    print("--- Đang phân tích lỗi... ---")
    analysis_state = error_analysis_agent.invoke(initial_state)
    print("✅ Phân tích hoàn tất.\n")

    # 4. Chuẩn bị cho vòng lặp hội thoại (Giai đoạn 2)
    current_state = analysis_state
    
    # --- QUAN TRỌNG: XỬ LÝ TRẠNG THÁI ĐẦU VÀO ---
    # prompt_crafter (trong bước phân tích) ĐÃ thêm bài làm đầu tiên vào history.
    # Vì vậy, ta cần xóa student_input tạm thời để node 'format_input' trong đồ thị 
    # không thêm nó vào lần nữa trong lần chạy đầu tiên.
    #current_state["student_input"] = "" 
    
    MAX_ROUNDS = 5 

    while current_state["round"] < MAX_ROUNDS:
        print(f"\n" + "-"*20 + f" Vòng {current_state['round'] + 1} " + "-"*20)
        
        # --- A. GIA SƯ AI ---
        print("👨‍🏫 Thầy Tích Hợp đang suy nghĩ...")
        
        # Gọi đồ thị hội thoại
        # Đồ thị sẽ tự động xử lý history thông qua các node bên trong
        output_state = tutoring_graph.invoke(current_state)
        
        # Lấy phản hồi của gia sư (tin nhắn cuối cùng trong history)
        # Lưu ý: Cấu trúc bây giờ là {'role': 'assistant', 'content': '...'}
        last_msg = output_state['conversation_history'][-1]
        tutor_message = last_msg['content']
        
        print(f"👨‍🏫 Thầy Tích Hợp: {tutor_message}")

        # Kiểm tra điều kiện dừng (Học sinh đã trả lời đúng chưa?)
        # Lưu ý: Logic check đúng sai cần đảm bảo evaluator_node đã chạy
        # (Trong code graph_conversation hiện tại của bạn evaluator đang tạm tắt hoặc chạy sau input,
        # hãy chắc chắn output_state có trường is_correct được cập nhật)
        
        # --- B. HỌC SINH AI (ITF SIMULATION) ---
        print("\n🧠 [ITF Internal Processing] Học sinh đang xử lý thông tin...")
        time.sleep(3) # Delay nhỏ tạo cảm giác thực
        print(f"Bloom level:{input_solution['bloom_level']}")
        raw_simulation = generate_simulated_student_response(
            conversation_history=output_state["conversation_history"],
            question_text=question_text,
            current_misconception=input_solution['explanation'],
            competency_level=input_solution['bloom_level'],
            motivation_level= "LOW"
        )
        
        sim_data = parse_simulation_output(raw_simulation)
        
        print(f"   💭 Suy nghĩ: {sim_data['thought']}")
        print(f"   ⚙️ Hành động: {sim_data['action']}")
        print(f"🧑‍🎓 An nói: {sim_data['response']}")
        if sim_data['action'] == "CORRECTING":
            save_conversation_json(output_state["conversation_history"])
            print("\n🎉 [HỆ THỐNG]: Student understand. Break")
            break
        # --- C. CẬP NHẬT TRẠNG THÁI CHO VÒNG SAU ---
        current_state = output_state
        
        # Cập nhật input mới của học sinh để node 'format_input' xử lý ở đầu vòng sau
        current_state["student_input"] = sim_data['response']
        
        # Tăng đếm vòng lặp
        current_state["round"] += 1
        
        # --- LƯU Ý QUAN TRỌNG ---
        # KHÔNG dùng lệnh append dưới đây nữa:
        # current_state["conversation_history"].append(...) 
        # Vì node 'format_input' trong đồ thị sẽ tự động làm việc này khi invoke lại.

    if current_state["round"] >= MAX_ROUNDS:
        save_conversation_json(output_state["conversation_history"])
        print("\n⚠️ Đã đạt giới hạn số vòng lặp. Kết thúc mô phỏng.")