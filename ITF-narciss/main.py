# /thay_tich_hop/main.py

from graph_initialization import create_error_analysis_agent
from graph_conversation import create_tutoring_graph
import warnings

# Tạm thời bỏ qua cảnh báo deprecation để output sạch hơn
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")


if __name__ == "__main__":
    # 1. Biên dịch các tác nhân/đồ thị khi khởi động
    error_analysis_agent = create_error_analysis_agent()
    tutoring_graph = create_tutoring_graph()

    # 2. Thiết lập trạng thái ban đầu đầy đủ
    initial_solution = "Mệnh đề đảo: 'Nếu tam giác ABC không cân thì tam giác ABC không đều'."
    
    initial_state = {
        "student_id": "An",
        "question_id": "QB_01_01_01",
        "initial_student_solution": initial_solution,
        "student_input": initial_solution, # Ban đầu, input coi như là bài làm
        "round": 0,
        "conversation_history": [],
        "is_correct": False,
        "is_final": False,
        # Các trường này sẽ được điền bởi đồ thị phân tích
        "instructional_context": {},
        "error_analysis": {},
        "learner_profile": {},
        "master_prompt": "",
    }

    # =======================================================================
    # GIAI ĐOẠN 1: Chạy tác nhân phân tích lỗi (CHỈ MỘT LẦN)
    # =======================================================================
    print("="*50 + "\nBẮT ĐẦU PHÂN TÍCH BÀI LÀM BAN ĐẦU...\n" + "="*50)
    analysis_state = error_analysis_agent.invoke(initial_state)
    print("\n" + "="*50 + "\nPHÂN TÍCH HOÀN TẤT. BẮT ĐẦU HỘI THOẠI DẠY KÈM...\n" + "="*50)

    # =======================================================================
    # GIAI ĐOẠN 2: Bắt đầu vòng lặp hội thoại
    # =======================================================================
    
    # Trạng thái hiện tại bắt đầu từ kết quả của giai đoạn phân tích
    current_state = analysis_state
    #print(f"\n--- Trạng thái ban đầu cho hội thoại dạy kèm: {current_state} ---\n")
    # Thêm bài làm ban đầu của học sinh vào lịch sử để gia sư có thể "nhìn thấy"
    # Điều này rất quan trọng cho lượt gọi invoke() đầu tiên của tutoring_graph
    #current_state["conversation_history"].append({'role': 'user', 'content': initial_solution})

    while True:
        # 1. Gọi đồ thị dạy kèm với trạng thái hiện tại
        # Lượt đầu tiên, nó sẽ bắt đầu từ entry_point và chạy evaluator -> teacher
        output_state = tutoring_graph.invoke(current_state)
        #print(f"\n--- Trạng thái sau khi gọi đồ thị dạy kèm: {output_state['conversation_history']} ---\n")
        # 2. Lấy câu hỏi của gia sư và hiển thị
        # === SỬA LỖI Ở ĐÂY ===
        # Thay thế 'parts' bằng 'content' để khớp với định dạng dữ liệu mới
        # 'content' là một chuỗi, không phải danh sách, nên không cần [0]
        tutor_message = output_state['conversation_history'][-1]['content']
        print(f"\nThầy Tích Hợp: {tutor_message}")

        # 3. Nếu câu trả lời trước đó là đúng, kết thúc
        # output_state['is_correct'] sẽ được cập nhật bởi evaluator_node
        if output_state.get('is_correct'):
            print("\n[HỆ THỐNG]: Tuyệt vời! Em đã trả lời đúng. Buổi học kết thúc.")
            break

        # 4. Nhận đầu vào mới từ người dùng
        try:
            user_input = input("An: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nTạm biệt!")
                break
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break

        # 5. Chuẩn bị trạng thái cho lượt tiếp theo
        # Bắt đầu với trạng thái đầu ra từ lượt vừa rồi
        current_state = output_state
        # Cập nhật các trường động cho lượt tiếp theo
        current_state["student_input"] = user_input
        # === SỬA LỖI Ở ĐÂY ===
        # Sử dụng 'content' khi thêm tin nhắn mới của người dùng
        #current_state["conversation_history"].append({'role': 'user', 'content': user_input})
        # Tăng số vòng lặp (tùy chọn, hữu ích cho việc debug)
        current_state["round"] += 1