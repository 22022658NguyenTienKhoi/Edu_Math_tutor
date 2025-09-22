# /educationq_project/main.py

import os
from typing import List

# Import from our custom modules
from data_models import QuestionItem
from workflow import EducationQWorkflow

def create_sample_questions() -> List[QuestionItem]:
    """Create sample questions for testing."""
    return [
        QuestionItem(
            id="q1",
            question="Nhiệt độ sôi của nước là bao nhiêu độ Celcius?",
            options=["10", "60", "100"],
            correct_answer=2,
            explanation="nước sôi ở 100 độ C"
        )
    ]

def main():
    """Main function to demonstrate the EducationQ system."""
    # Set up API key
    import dotenv
    dotenv.load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

    # Create sample questions
    questions = create_sample_questions()

    # Initialize and run the workflow
    education_q = EducationQWorkflow(
        model_name='models/gemma-3-27b-it',
        student_model_name='models/gemma-3-12b-it'
    )
    results = education_q.run(questions)

    # Display results
    print("\n" + "="*50)
    print("EDUCATIONQ ASSESSMENT RESULTS")
    print("="*50)
    # Pretty print the results
    import json
    print(json.dumps(results['summary'], indent=2))


if __name__ == "__main__":
    main()