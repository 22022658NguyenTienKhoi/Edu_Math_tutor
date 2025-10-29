# /education_project/tools.py

from typing import Dict, Any

def query_knowledge_base(query: str) -> str:
    """
    PLACEHOLDER: Giả lập việc truy vấn vào một cơ sở dữ liệu vector (RAG).
    Trong thực tế, hàm này sẽ kết nối đến Pinecone, ChromaDB, etc.
    """
    print(f"--- TOOL CALL (Placeholder): Querying Knowledge Base for '{query}' ---")
    if "diện tích hình chữ nhật" in query.lower():
        return "Kiến thức từ SGK: Diện tích hình chữ nhật được tính bằng cách lấy chiều dài nhân với chiều rộng."
    if "định lý pythagore" in query.lower():
        return "Kiến thức từ SGK: Trong một tam giác vuông, bình phương cạnh huyền bằng tổng bình phương của hai cạnh góc vuông (a^2 + b^2 = c^2)."
    return "Kiến thức từ SGK: Không tìm thấy thông tin liên quan."

def get_student_profile(student_id: str) -> Dict[str, Any]:
    """
    PLACEHOLDER: Giả lập việc lấy thông tin hồ sơ học tập của học sinh.
    Trong thực tế, hàm này sẽ truy vấn vào DB người dùng.
    """
    print(f"--- TOOL CALL (Placeholder): Fetching profile for student '{student_id}' ---")
    # Hồ sơ này cho thấy học sinh thường gặp khó khăn với các bài toán có nhiều bước.
    return {
        "name": "Alex",
        "learning_history": ["Struggles with multi-step problems", "Strong in basic arithmetic"],
        "preferred_style": "no information"
    }

def log_to_monitoring_service(data: Dict[str, Any]):
    """
    PLACEHOLDER: Giả lập việc ghi log ra một hệ thống giám sát.
    Trong thực tế, có thể gửi dữ liệu tới Datadog, Grafana, etc.
    """
    print(f"--- TOOL CALL (Placeholder): Logging to Monitoring Service ---")
    print(data)