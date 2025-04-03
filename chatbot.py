# Import necessary libraries
import streamlit as st  # For creating the web app interface
from langchain_ollama.llms import OllamaLLM # To connect to Ollama models
from langchain_core.prompts import ChatPromptTemplate # To structure the prompt sent to the LLM
from langchain_core.output_parsers import StrOutputParser # To get just the text response

# --- Basic App Configuration ---

# Set the title that appears at the top of the web page
st.title("💬 My Simple AI Chatbot (Phase 1)")
# Updated caption to reflect the new model
st.caption("Using Ollama (llama3.2) and LangChain")

# --- Connect to the Local LLM (Ollama) ---

# Initialize a connection to the Ollama model.
# Make sure Ollama is running and you have pulled the 'llama3.2' model.
# If you pulled a different model, change 'llama3.2' below to match.
try:
    # *** MODIFIED LINE: Changed model name from "phi3:mini" to "llama3.2" ***
    llm = OllamaLLM(model="llama3.2")
except Exception as e:
    # Updated error message
    st.error(f"Could not connect to Ollama. Is it running? Have you pulled 'llama3.2'? Error: {e}")
    st.stop() # Stop the app if connection fails

# --- Define the Prompt Structure ---

# Create a template for how we'll ask the LLM questions.
# This helps give the LLM context.
# "{user_question}" is a placeholder where the user's input will go.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question concisely."),
    ("human", "{user_question}") # Placeholder for the input from the text box
])

# --- Create the Processing Chain ---

# This sets up the sequence:
# 1. Take the user's question and format it using the prompt_template.
# 2. Send the formatted prompt to the llm (Ollama model).
# 3. Parse the output to get a clean string response (StrOutputParser).
chain = prompt_template | llm | StrOutputParser()

# --- Create the User Interface Elements ---

# Create a text input box where the user can type their question.
# The text typed by the user will be stored in the 'user_input' variable.
user_input = st.text_input("Ask something:", key="user_query")

# --- Handle User Input and Display Response ---

# Check if the user has typed something in the box and pressed Enter.
if user_input:
    # If there's input, display a "Thinking..." message while processing.
    with st.spinner("Thinking..."):
        # Send the user's input through the processing chain.
        # The 'invoke' method runs the chain. We pass the input in a dictionary
        # matching the placeholder name in our prompt_template.
        response = chain.invoke({"user_question": user_input})

        # Display the response received from the LLM.
        st.write("Assistant:", response)

