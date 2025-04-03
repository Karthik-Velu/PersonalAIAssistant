# Import necessary libraries
import streamlit as st
# *** MODIFIED IMPORT: Using ChatGroq instead of Google/Ollama ***
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os # To potentially load API key locally (optional)

# --- Basic App Configuration ---
st.title("💬 My Conversational AI Chatbot (Groq Version)")
# Updated caption
st.caption("Using Groq (Llama 3 8B), LangChain, and Streamlit Session State")

# --- Load API Key ---
# Try to get the API key from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

# Optional: Fallback for local development (if you want to run locally too)
# You might need to set the GROQ_API_KEY environment variable locally for this.
# if not groq_api_key:
#     try:
#         groq_api_key = os.environ.get("GROQ_API_KEY")
#     except Exception:
#         pass # Handle case where environment variable is not set

if not groq_api_key:
    st.error("Groq API Key not found. Please add it to Streamlit secrets (key: GROQ_API_KEY).")
    st.stop()

# --- Setup Groq Connection ---
try:
    # *** MODIFIED LLM Initialization: Using ChatGroq ***
    # Common models available on Groq include:
    # llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it
    # Check Groq's documentation for the latest available models.
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192" # Using Llama 3 8B as an example
        )
except Exception as e:
    st.error(f"Could not initialize Groq Chat model. Error: {e}")
    st.stop()

# --- Setup Conversation Memory ---
# (This part remains the same as the previous version)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Define the Prompt Structure ---
# (This part remains the same)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's questions based on the conversation history."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# --- Create the Conversation Chain ---
# (This part remains the same, just uses the new 'llm' object)
conversation_chain = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=False # Set to True to see chain details in terminal (optional)
)

# --- Display Existing Chat History ---
# (This part remains the same)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
# (This part remains the same)
if user_input := st.chat_input("Ask something:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking... (Using Groq)"): # Updated spinner message
        try:
            response = conversation_chain.predict(input=user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred while getting the response from Groq: {e}")
            # Optionally remove the user message if processing failed badly
            # if st.session_state.messages[-1]["role"] == "user":
            #     st.session_state.messages.pop()

