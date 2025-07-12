import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Page config
st.set_page_config(
    page_title="AI ML Tutor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        padding-top: 40px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #3366cc;'>🤖 AI-Powered ML Code Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI Tutor for Machine Learning Implementations</p>", unsafe_allow_html=True)
st.markdown("---")

# Select algorithm
algo = st.selectbox("📚 Select a Machine Learning Algorithm:", 
                    ["LogisticRegression", "SVM", "SVC", "RandomForestClassifier", "Xgboost", "DecisionTreeClassifier"])

# Load API key
with open("E:/LANGCHAIN/keys/key.txt") as f:
    GOOGLE_API_KEY = f.read().strip()

# Chat model setup
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI Tutor with expertise in Data Science and AI who tells step-by-step Python implementation for Machine Learning Algorithms asked by user."),
    ("human", "Tell me a Python implementation for {topic_name}?"),
])

output_parser = StrOutputParser()
chain = chat_prompt_template | chat_model | output_parser

# Button
if st.button("🔍 See the Implementation"):
    with st.spinner(f"Generating implementation for **{algo}**..."):
        time.sleep(1.2)
        result = chain.invoke({"topic_name": algo})
        st.success("Here's your code! 👇")
        st.code(result, language="python")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Built with ❤️ by your AI Assistant & LangChain + Gemini APIs</p>", 
    unsafe_allow_html=True
)
