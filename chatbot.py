# Import necessary libraries
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
# Import specific message types from LangChain for MCP mapping
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # Add ToolMessage later if needed

# Import MongoDB specific library
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import os
import datetime
import urllib.parse
import json # For handling potential JSON content or metadata
import uuid # For potential unique event IDs

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("💬 Personalized AI Chatbot (MCP Structure + MongoDB)")
st.caption("Using Groq, LangChain, MongoDB Atlas (MCP Format), and Streamlit Session State")

# --- MongoDB Initialization --- (Same as mongo_v1)
@st.cache_resource
def initialize_mongo_client():
    """Initializes and returns the MongoDB client."""
    try:
        mongo_user = st.secrets.get("MONGO_USER")
        mongo_password = st.secrets.get("MONGO_PASSWORD")
        mongo_cluster_uri = st.secrets.get("MONGO_CLUSTER_URI")
        if not all([mongo_user, mongo_password, mongo_cluster_uri]):
            st.error("MongoDB credentials not found in Streamlit secrets.")
            st.stop()
        encoded_user = urllib.parse.quote_plus(mongo_user)
        encoded_password = urllib.parse.quote_plus(mongo_password)
        uri = f"mongodb+srv://{encoded_user}:{encoded_password}@{mongo_cluster_uri}/?retryWrites=true&w=majority"
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping') # Verify connection
        st.success("Pinged MongoDB deployment. Successfully connected!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize MongoDB client: {e}")
        st.stop()

mongo_client = initialize_mongo_client()
DB_NAME = st.secrets.get("MONGO_DB_NAME", "chatbot_db")
# Using a new collection name for MCP structured data
COLLECTION_NAME = st.secrets.get("MONGO_COLLECTION_NAME", "mcp_context_events")
try:
    db = mongo_client[DB_NAME]
    mcp_collection = db[COLLECTION_NAME]
    # Index for efficient querying by user and time
    mcp_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)])
    mcp_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
except Exception as e:
    st.error(f"Failed to access MongoDB database/collection or create index: {e}")
    st.stop()

# --- User Identification (Basic Example) --- (Same as mongo_v1)
st.sidebar.header("User Session")
user_id_input = st.sidebar.text_input("Enter a User ID for this session:", key="user_id_input")
if user_id_input:
    st.session_state.user_id = user_id_input
else:
    if "user_id" not in st.session_state:
        st.warning("Please enter a User ID to load/save history.")
        st.stop()
st.sidebar.markdown(f"**Current User ID:** `{st.session_state.user_id}`")
user_id = st.session_state.user_id

# --- MCP Structure & MongoDB Interaction Functions ---

def create_mcp_event(user_id, role, content, model_name=None, metadata=None):
    """Creates a dictionary representing an MCP-like event."""
    event = {
        "user_id": user_id,
        "event_id": str(uuid.uuid4()), # Generate a unique ID for the event
        "role": role, # "user", "assistant", "system", "tool"
        "content": content, # Can be string or structured data
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "metadata": metadata or {}, # Store extra info like model used
        # Placeholders for future extensions based on MCP spec
        "tool_calls": None,
        "tool_results": None,
    }
    if model_name and role == "assistant":
         event["metadata"]["model_used"] = model_name
    return event

def save_mcp_event_to_mongo(mcp_event):
    """Saves an MCP event document to MongoDB."""
    if not mcp_event.get("user_id"):
        st.warning("Cannot save event without User ID.")
        return
    try:
        mcp_collection.insert_one(mcp_event)
    except Exception as e:
        st.error(f"Failed to save MCP event to MongoDB: {e}")

def load_mcp_history_from_mongo(user_id, limit=50):
    """Loads the last 'limit' MCP events from MongoDB for a user."""
    if not user_id:
        return []
    try:
        cursor = mcp_collection.find({"user_id": user_id}).sort("timestamp", pymongo.DESCENDING).limit(limit)
        mcp_events_db = list(cursor)
        # Reverse to get chronological order (oldest first)
        return mcp_events_db[::-1]
    except Exception as e:
        st.error(f"Failed to load MCP history from MongoDB: {e}")
        return []

# --- Model Selection --- (Same as before)
AVAILABLE_GROQ_MODELS = [
    "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it",
]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = AVAILABLE_GROQ_MODELS[0]
selected_model_name = st.sidebar.selectbox(
    "Choose a Groq model:", options=AVAILABLE_GROQ_MODELS, key="model_selector",
    index=AVAILABLE_GROQ_MODELS.index(st.session_state.selected_model)
)
st.sidebar.markdown(f"**Current Model:** `{selected_model_name}`")

# --- Load API Key --- (Same as before)
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key: st.error("Groq API Key not found."); st.stop()

# --- Initialize LLM, Memory, and Chain (MCP Adapted) ---
def initialize_or_get_chain_mcp(model_name, user_id):
    chain_key = f"conversation_chain_{user_id}_{model_name}"
    memory_key = f"memory_{user_id}_{model_name}"
    messages_key = f"messages_{user_id}" # UI messages tied to user

    # Reset if model changed (keeps things simpler for now)
    if st.session_state.get("current_model_for_user_" + user_id) != model_name:
        if chain_key in st.session_state: del st.session_state[chain_key]
        if memory_key in st.session_state: del st.session_state[memory_key]
        st.session_state["current_model_for_user_" + user_id] = model_name

    # Initialize if not present in session state
    if chain_key not in st.session_state:
        st.info(f"Initializing chain for model: {model_name}")
        try:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        except Exception as e: st.error(f"Groq init error: {e}"); st.stop()

        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load history from MongoDB (MCP events)
        loaded_mcp_events = load_mcp_history_from_mongo(user_id)

        # *** CRITICAL MCP ADAPTATION: Populate LangChain memory from MCP events ***
        ui_messages = []
        for event in loaded_mcp_events:
            role = event.get("role")
            content = event.get("content", "")
            # Map MCP roles/content to LangChain message objects
            if role == "user":
                memory.chat_memory.add_message(HumanMessage(content=content))
                ui_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                memory.chat_memory.add_message(AIMessage(content=content))
                ui_messages.append({"role": "assistant", "content": content})
            elif role == "system": # Handle system messages if you store them
                 memory.chat_memory.add_message(SystemMessage(content=content))
                 # Optionally add system messages to UI or handle differently
            # Add handling for "tool" role later if needed

        st.session_state[memory_key] = memory
        st.session_state[messages_key] = ui_messages # Set UI messages based on loaded history

        # Define prompt (can now potentially include profile loaded from DB)
        # <<< PLACEHOLDER: Load User Profile from DB >>>
        # user_profile_summary = load_user_profile(user_id) # Function to implement
        # profile_prompt = f"User Profile Notes: {user_profile_summary}" if user_profile_summary else ""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant running on {model_name}. Use history. User ID: {user_id}."),
            # ("system", profile_prompt), # Inject profile if loaded
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = ConversationChain(llm=llm, prompt=prompt_template, memory=st.session_state[memory_key], verbose=False)
        st.session_state[chain_key] = chain

    return st.session_state[chain_key]

# Get the chain for the current user and selected model
conversation_chain = initialize_or_get_chain_mcp(selected_model_name, user_id)
messages_key = f"messages_{user_id}" # Key for UI messages

# Initialize UI message list if it wasn't populated by loading history
if messages_key not in st.session_state:
     st.session_state[messages_key] = []

# --- Display Existing Chat History (from session state) ---
for message in st.session_state[messages_key]:
    role = message.get("role")
    content = message.get("content")
    if role and content:
         with st.chat_message(role): st.markdown(content)

# --- Handle User Input ---
if user_input := st.chat_input("Ask something:"):
    # Add user message to UI display state
    st.session_state[messages_key].append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    # Create and save user MCP event
    user_mcp_event = create_mcp_event(user_id, "user", user_input)
    save_mcp_event_to_mongo(user_mcp_event) # SAVE USER EVENT

    # <<< PLACEHOLDER: Prompt Enhancement Logic (using profile) >>>
    final_input = user_input # Using original input for now

    # Get response using the current chain
    with st.spinner(f"Thinking... (Using {selected_model_name})"):
        try:
            # <<< PLACEHOLDER: Actual Answer Generation >>>
            response = conversation_chain.predict(input=final_input) # This automatically updates LangChain memory

            # Add assistant response to UI display state
            st.session_state[messages_key].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"): st.markdown(response)
            # Create and save assistant MCP event
            assistant_mcp_event = create_mcp_event(user_id, "assistant", response, model_name=selected_model_name)
            save_mcp_event_to_mongo(assistant_mcp_event) # SAVE AI EVENT

            # <<< PLACEHOLDER: Profile Update Logic >>>
            # update_user_profile_async(user_id, user_mcp_event, assistant_mcp_event)

        except Exception as e:
            st.error(f"An error occurred: {e}")


# <<< PLACEHOLDER: Gmail Integration Logic >>>
# st.sidebar.button("Connect Gmail (Placeholder)")