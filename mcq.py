import streamlit as st
import pandas as pd
import os
import base64
from utils import QuestionGenerator, QuizManager

# Function to load and encode images for background
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# Set page configuration
st.set_page_config(
    page_title="NIELIT Quiz Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean modern UI styling
st.markdown("""
    <style>
        * {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .main {
            background-color: #f8f9fa;
            padding: 0;
        }
        [data-testid="stSidebar"] {
            background-color: #1e3a8a;
            padding-top: 2rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] label {
            color: white;
        }
        .content-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        h1 {
            color: #1e3a8a;
            font-weight: 700;
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        h2 {
            color: #1e3a8a;
            font-weight: 600;
            font-size: 1.5rem;
            margin-bottom: 0.75rem;
        }
        .question-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .question-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        }
        .question-number {
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 8px;
            font-size: 1.1rem;
            display: inline-block;
            padding: 2px 8px;
            background-color: #f0f7ff;
            border-radius: 6px;
        }
        .question-text {
            font-size: 1.05rem;
            line-height: 1.5;
            margin-top: 5px;
            color: #333;
        }
        .correct-answer {
            background-color: #f0fdf4;
            border-left: 5px solid #10b981;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 16px 0;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
            position: relative;
            color: #1f1987;
        }
        .correct-answer::after {
            content: "✓";
            position: absolute;
            top: 12px;
            right: 15px;
            font-size: 1.4rem;
            color: #1f1987;
            font-weight: bold;
        }
        .incorrect-answer {
            background-color: #fef2f2;
            border-left: 5px solid #ef4444;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 16px 0;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
            position: relative;
            color: #1f1987;
        }
        .incorrect-answer::after {
            content: "✗";
            position: absolute;
            top: 12px;
            right: 15px;
            font-size: 1.4rem;
            color: #991b1b;
            font-weight: bold;
        }
        .result-question {
            font-size: 1.1rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 10px;
        }
        .answer-detail {
            margin-top: 8px;
            padding: 8px 12px;
            background-color: rgba(255, 255, 255, 1);
            border-radius: 8px;
            color: #1f1987;
            font-size: 1rem;
        }
        .score-display {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
            position: relative;
            overflow: hidden;
        }
        .score-display::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            opacity: 0.6;
        }
        .score-percentage {
            position: relative;
            font-size: 3rem;
            font-weight: 800;
            margin: 10px 0;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .score-label {
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .score-fraction {
            font-size: 1.2rem;
            font-weight: 500;
            position: relative;
        }
        .app-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        .stButton > button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
            box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3);
            transform: translateY(-2px);
        }
        .submit-button .stButton > button {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            box-shadow: 0 4px 10px rgba(16, 185, 129, 0.2);
        }
        .submit-button .stButton > button:hover {
            background: linear-gradient(90deg, #059669 0%, #047857 100%);
            box-shadow: 0 6px 15px rgba(16, 185, 129, 0.3);
        }
        .stSelectbox > div > div, 
        .stNumberInput > div > div:last-child {
            border-radius: 5px;
        }
        button[kind="header"] {
            display: none;
        }
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 20px;
        }
        .stRadio > label {
            padding: 10px 15px !important;
            border-radius: 8px !important;
            border: 1px solid #e2e8f0 !important;
            width: 100% !important;
            display: block !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease !important;
        }
        .stRadio > label:hover {
            background-color: #f8fafc !important;
            border-color: #cbd5e1 !important;
        }
        .stTextInput > div > div {
            border-radius: 8px !important;
        }
        .stTextInput input {
            border-radius: 8px !important;
            padding: 10px 12px !important;
            border-color: #e2e8f0 !important;
        }
        .stTextInput input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'quiz_manager' not in st.session_state:
    st.session_state.quiz_manager = QuizManager()
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: white; text-align: center;">NIELIT Quiz</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0aec0; text-align: center; margin-bottom: 20px;">Configure your quiz settings</p>', unsafe_allow_html=True)
    
    # Source selection
    st.markdown('<p style="color: #a0aec0; font-weight: 600;">Question Source</p>', unsafe_allow_html=True)
    source = st.selectbox("", ["QuanBank", "AutoGen"], index=0, label_visibility="collapsed")
    
    # Module selection for QuanBank
    module = None
    topic = "Module-Based"  # Default for QuanBank
    question_type = "Multiple Choice"  # Default for QuanBank
    if source == "QuanBank":
        st.markdown('<p style="color: #a0aec0; font-weight: 600; margin-top: 15px;">Module</p>', unsafe_allow_html=True)
        module1 = st.selectbox("", ["1. Python Programming", "2. Statistical Concepts", "3. Data Science and Analytics", "4. Machine Learning Mandatory", "5. Deep Learning"], index=0, label_visibility="collapsed")
        module = module1.split(".")[0].strip()
        topic = f"Module {module}" if module else "Module-Based"
    else:
        # Quiz configuration for AutoGen only
        st.markdown('<p style="color: #a0aec0; font-weight: 600; margin-top: 15px;">Question Format</p>', unsafe_allow_html=True)
        question_type = st.selectbox("", ["Multiple Choice", "Fill in the Blank"], index=0, label_visibility="collapsed")
        
        st.markdown('<p style="color: #a0aec0; font-weight: 600; margin-top: 15px;">Subject Area</p>', unsafe_allow_html=True)
        topic = st.text_input("Enter a topic (e.g. Operating System, DBMS)", value="")

    
    # Common settings
    st.markdown('<p style="color: #a0aec0; font-weight: 600; margin-top: 15px;">Difficulty Level</p>', unsafe_allow_html=True)
    difficulty = st.selectbox("", ["Easy", "Medium", "Hard"], index=1, label_visibility="collapsed")
    
    st.markdown('<p style="color: #a0aec0; font-weight: 600; margin-top: 15px;">Number of Questions</p>', unsafe_allow_html=True)
    num_questions = st.number_input("", min_value=1, max_value=30, value=5, label_visibility="collapsed")
    
    # Generate quiz button
    st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
    generate_quiz = st.button("Generate Quiz", use_container_width=True)
    
    
    # Sidebar footer
    st.markdown('<div class="footer" style="color: #a0aec0; margin-top: 40px;">NIELIT MCQ Generator<br>© 2025</div>', unsafe_allow_html=True)

# Main content
content_col = st.container()

with content_col:
    st.markdown('<div class="content-card app-header">', unsafe_allow_html=True)
    st.markdown('''
    <div style="text-align: center;">
        <div style="display: inline-block; background: linear-gradient(90deg, #1e40af, #3b82f6); padding: 8px 16px; border-radius: 30px; margin-bottom: 15px;">
            <span style="color: white; font-weight: 700; letter-spacing: 1.5px;">NIELIT</span>
        </div>
        <h1 style="text-align: center; margin-top: 5px;">MCQ Generator & Quiz Platform</h1>
        <p style="text-align: center; max-width: 600px; margin: 10px auto; color: #4b5563;">
            Create customized quizzes on various computer science topics to enhance your learning experience
        </p>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process quiz generation
    if generate_quiz:
        if topic.strip() == "":
            st.error("Please enter a topic before generating the quiz.")

        else:
            with st.spinner("Creating your personalized quiz..."):
                st.session_state.quiz_submitted = False
                generator = QuestionGenerator()
                st.session_state.quiz_generated = st.session_state.quiz_manager.generate_questions(
                    generator, topic, question_type, difficulty, num_questions, source, module
                )
                st.rerun()

    # Display quiz if generated
    if st.session_state.quiz_generated and st.session_state.quiz_manager.questions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        display_topic = st.session_state.quiz_manager.current_topic
        st.markdown(f'''
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h2 style="margin: 0;">{display_topic} Quiz</h2>
            <div style="background-color: #e0f2fe; color: #0369a1; font-weight: 600; padding: 5px 12px; border-radius: 20px; font-size: 0.9rem;">
                {difficulty} Level
            </div>
        </div>
        <p>Complete all {num_questions} questions and submit your answers. For multiple choice questions, select exactly one option.</p>
        <div style="height: 3px; width: 100px; background: linear-gradient(90deg, #3b82f6, #93c5fd); margin: 15px 0;"></div>
        ''', unsafe_allow_html=True)
        
        st.session_state.quiz_manager.attempt_quiz()
        
        st.markdown('<div class="submit-button">', unsafe_allow_html=True)
        submit_quiz = st.button("Submit Quiz", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit_quiz:
            for i, q in enumerate(st.session_state.quiz_manager.questions):
                if q['type'] == 'MCQ':
                    radio_key = f"mcq_radio_{st.session_state.quiz_manager.current_quiz_id}_{i}"
                    selected_option = st.session_state.get(radio_key, "")
                    st.session_state.quiz_manager.user_answers[i] = selected_option
                else:
                    st.session_state.quiz_manager.user_answers[i] = st.session_state[f"fill_blank_{st.session_state.quiz_manager.current_quiz_id}_{i}"]
            
            st.session_state.quiz_manager.evaluate_quiz()
            st.session_state.quiz_submitted = True
            st.rerun()

    # Display results if quiz submitted
    if st.session_state.quiz_submitted:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<h2>Quiz Results</h2>', unsafe_allow_html=True)
        
        results_df = st.session_state.quiz_manager.generate_result_dataframe()
        
        if not results_df.empty:
            correct_count = results_df['is_correct'].sum()
            total_questions = len(results_df)
            score_percentage = (correct_count / total_questions) * 100
            
            st.markdown(f"""
            <div class="score-display">
                <div class="score-label">Your Score</div>
                <div class="score-percentage">{score_percentage:.1f}%</div>
            <div class="score-fraction">{correct_count} out of {total_questions} correct</div>
            </div>
            """, unsafe_allow_html=True)
            
            for _, result in results_df.iterrows():
                if result['is_correct']:
                    st.markdown(f"""
                    <div class="correct-answer">
                        <div class="result-question">Question {result['question_number']}</div>
                        <p>{result['question']}</p>
                        <div class="answer-detail">
                            <strong>Your Answer:</strong> {result['user_answer']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="incorrect-answer">
                        <div class="result-question">Question {result['question_number']}</div>
                        <p>{result['question']}</p>
                        <div class="answer-detail">
                            <strong>Your Answer:</strong> {result['user_answer']}
                        </div>
                        <div class="answer-detail" style="background-color: rgba(16, 185, 129, 0.1);">
                            <strong>Correct Answer:</strong> {result['correct_answer']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
        else:
            st.warning("No results available. Please complete the quiz first.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Welcome screen if no quiz is generated yet
    if not st.session_state.quiz_generated and not st.session_state.quiz_submitted:
        st.markdown('<div class="content-card" style="text-align: center; padding: 30px;">', unsafe_allow_html=True)
        st.markdown('<h2>Welcome to the NIELIT MCQ Generator</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1rem; margin-bottom: 20px;">Create customized quizzes to test your knowledge in various computer science subjects.</p>
        <div style="display: flex; justify-content: center; margin: 30px 0;">
            <div style="max-width: 500px; text-align: left;">
                <p style="font-weight: 600; margin-bottom: 10px;">How to use:</p>
                <ol>
                    <li>Select your preferred settings from the sidebar</li>
                    <li>Click "Generate Quiz" to create questions</li>
                    <li>Answer the questions and submit your responses</li>
                </ol>
            </div>
        </div>
        <p style="font-style: italic; color: #4b5563;">Perfect for exam preparation and self-assessment</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)