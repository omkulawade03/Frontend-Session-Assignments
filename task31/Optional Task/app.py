import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Dual-Mode Agentic AI", page_icon="🤖", layout="centered")

st.title("🤖 Dual-Mode Agentic AI Chatbot")
st.caption("Powered by LangChain & Hugging Face")

# Sidebar for configuration
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Select Model Mode:",
    ["API Mode (DeepSeek-R1)", "Local Mode (TinyLlama)"]
)

# Caching model instances to prevent reloading on every interaction
@st.cache_resource
def load_api_model():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        max_new_tokens=256,
        temperature=0.7,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm)

@st.cache_resource
def load_local_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 150, "do_sample": True, "temperature": 0.7},
    )
    return ChatHuggingFace(llm=llm)

# Initialize chosen model
if mode == "API Mode (DeepSeek-R1)":
    chat_model = load_api_model()
    if chat_model is None:
        st.error("Error: HUGGINGFACEHUB_API_TOKEN is missing in your .env file!")
        st.stop()
else:
    with st.spinner("Loading local model into memory..."):
        chat_model = load_local_model()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt_input := st.chat_input("Type your message..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Build Prompt Template Chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a very polite, kind, and helpful AI assistant."),
        ("user", "{input}")
    ])
    chain = prompt_template | chat_model

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"input": prompt_input})
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"Error generating response: {e}")