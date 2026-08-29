import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Streamlit Page Setup
st.set_page_config(page_title="Domain Knowledge RAG Chatbot", page_icon="🤖")
st.title("🤖 Knowledge-Base RAG Chatbot")
st.subheader("Ask questions restricted strictly to internal documentation.")

# Initialize API Keys if not set in environment
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.sidebar.text_input("Enter Groq API Key:", type="password")

# Load Retriever and Model (Cached to prevent reloading on re-renders)
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="chroma-db",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    # Initialize LLM via init_chat_model
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    return retriever, llm

# System Prompt restricting answers strictly to retrieved context
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not contained in the context, say 'I could not find the answer.' "
    "Do not make up information.\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Input
if user_question := st.chat_input("Ask a question about company policy..."):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please provide a Groq API Key to continue.")
        st.stop()
        
    # Append user question to UI
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Retrieving knowledge base and generating answer..."):
        try:
            retriever, llm = load_rag_pipeline()
            
            # 1. Retrieve Context
            docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Construct Prompt & Generate Answer
            formatted_prompt = prompt_template.format_messages(context=context, input=user_question)
            response = llm.invoke(formatted_prompt)
            answer = response.content

            # 3. Display Assistant Answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Expandable view to inspect grounded context source
                with st.expander("View Retrieved Context Chunks"):
                    st.write(context if context else "No relevant context found.")
            
            # Store assistant response in history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error generating response: {e}")