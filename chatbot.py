# Import necessary libraries
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import MongoDB specific library
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

import os
import datetime
import urllib.parse
import json
import uuid

# Import Streamlit component lib for JS communication
import streamlit.components.v1 as components

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("️️🎙️ Conversational AI Chatbot (Speech Enabled)")
st.caption("Using Groq, LangChain, MongoDB Atlas (MCP Format), Web Speech API")

# --- Constants ---
MUFASA_VOICE_NAME = "Mufasa-like (Best Effort)"
DEFAULT_VOICE_NAME = "Default"
MUFASA_IMAGE_URL = "https://placehold.co/200x200/E9D5A1/A0522D?text=Lion+King+Mode" # Placeholder image
# Define available models list globally
AVAILABLE_GROQ_MODELS = [
    "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it",
]

# --- MongoDB Initialization ---
@st.cache_resource
def initialize_mongo_client():
    """Initializes and returns the MongoDB client."""
    # Wrapped in try-except for robustness during initialization
    try:
        mongo_user = st.secrets.get("MONGO_USER")
        mongo_password = st.secrets.get("MONGO_PASSWORD")
        mongo_cluster_uri = st.secrets.get("MONGO_CLUSTER_URI")
        if not all([mongo_user, mongo_password, mongo_cluster_uri]):
            st.error("MongoDB credentials (MONGO_USER, MONGO_PASSWORD, MONGO_CLUSTER_URI) not found in Streamlit secrets.")
            st.stop() # Stop if secrets are missing
        encoded_user = urllib.parse.quote_plus(mongo_user)
        encoded_password = urllib.parse.quote_plus(mongo_password)
        uri = f"mongodb+srv://{encoded_user}:{encoded_password}@{mongo_cluster_uri}/?retryWrites=true&w=majority"
        # Added timeout, adjust as needed
        client = MongoClient(uri, server_api=ServerApi('1'), connectTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth. Verifies connection.
        client.admin.command('ping')
        # st.success("MongoDB connected!") # Reduce verbose output in production
        return client
    except pymongo.errors.ConfigurationError as e:
         st.error(f"MongoDB Configuration Error: {e}. Check connection string and credentials format in Streamlit Secrets.")
         st.stop()
    except pymongo.errors.OperationFailure as e:
         st.error(f"MongoDB Authentication Error: {e}. Check database username/password in Streamlit Secrets and Network Access rules in Atlas.")
         st.stop()
    except Exception as e: # Catch other potential errors during init
        st.error(f"Failed to initialize MongoDB client: {e}")
        st.stop()

mongo_client = initialize_mongo_client()
# Use secrets for DB and Collection names, with defaults
DB_NAME = st.secrets.get("MONGO_DB_NAME", "chatbot_db")
COLLECTION_NAME = st.secrets.get("MONGO_COLLECTION_NAME", "mcp_context_events")
try:
    db = mongo_client[DB_NAME]
    mcp_collection = db[COLLECTION_NAME]
    # Ensure index exists for efficient querying by user and time (descending for latest)
    mcp_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
except Exception as e:
    st.error(f"Failed to access MongoDB collection '{COLLECTION_NAME}' or create index: {e}")
    st.stop() # Stop if DB access fails after connection

# --- User Identification ---
st.sidebar.header("User Session")
# Initialize user_id in session state if it doesn't exist
if "user_id" not in st.session_state:
    st.session_state.user_id = ""

user_id_input = st.sidebar.text_input(
    "Enter a User ID:",
    value=st.session_state.user_id, # Use state for persistence within session
    key="user_id_input_widget" # Use a key for the widget itself
)
# Update session state only if input changes
if user_id_input != st.session_state.user_id:
     st.session_state.user_id = user_id_input
     # Clear session data related to the previous user when ID changes
     keys_to_clear = [k for k in st.session_state if st.session_state.user_id in k and k != "user_id"] # Check if OLD user_id is in key name
     for key in keys_to_clear:
         if key in st.session_state: del st.session_state[key]
     st.rerun() # Rerun to reload data/state for the new user

# Ensure user_id is not empty before proceeding
user_id = st.session_state.user_id
if not user_id:
    st.warning("Please enter a User ID to begin.")
    st.stop() # Stop execution if no user ID is provided
st.sidebar.markdown(f"**User ID:** `{user_id}`")


# --- MCP Structure & MongoDB Functions ---
def create_mcp_event(user_id, role, content, model_name=None, metadata=None):
    """Creates a dictionary representing an MCP-like event."""
    # *** FIX: Removed invalid /* ... */ comment syntax ***
    event = {
        "user_id": user_id,
        "event_id": str(uuid.uuid4()), # Unique ID for each event
        "role": role, # "user", "assistant", "system", "tool"
        "content": content, # The actual message or data
        "timestamp": datetime.datetime.now(datetime.timezone.utc), # Use UTC for consistency
        "metadata": metadata or {}, # Store extra info like model used, latency, etc.
        "tool_calls": None, # Placeholder for future tool use
        "tool_results": None, # Placeholder for future tool use
    }
    # Add model name to metadata if it's an assistant message
    if model_name and role == "assistant":
        event["metadata"]["model_used"] = model_name
    return event

def save_mcp_event_to_mongo(mcp_event):
    """Saves an MCP event document to MongoDB."""
    if not mcp_event.get("user_id"):
        st.warning("Attempted to save event without User ID.")
        return # Don't save if user_id is missing
    try:
        # Use the globally accessible mcp_collection defined after client initialization
        mcp_collection.insert_one(mcp_event)
    except Exception as e:
        st.error(f"Failed to save event to MongoDB: {e}") # Log error but don't stop app

def load_mcp_history_from_mongo(user_id, limit=50):
    """Loads the last 'limit' MCP events from MongoDB for a user."""
    if not user_id: return [] # Return empty list if no user ID
    try:
        # Use the globally accessible mcp_collection
        # Sort by timestamp descending to get the latest messages, then limit
        cursor = mcp_collection.find({"user_id": user_id}).sort("timestamp", pymongo.DESCENDING).limit(limit)
        messages_db = list(cursor)
        # Reverse the list to get chronological order (oldest first) for memory/display
        return messages_db[::-1]
    except Exception as e:
        st.error(f"Failed to load history from MongoDB: {e}")
        return [] # Return empty list on error


# --- State Initialization --- (Ensure all keys are initialized before use)
# Moved AVAILABLE_GROQ_MODELS definition earlier
default_states = {
    "session_initialized": False,
    "selected_voice_name": DEFAULT_VOICE_NAME,
    "tts_voices": [],
    "stt_output": "",
    "stt_listening_toggle": False,
    "last_stt_processed": None,
    "selected_model_name": AVAILABLE_GROQ_MODELS[0], # Use defined list
    "text_to_speak_trigger": None,
    "last_spoken_content": ""
}
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# Initialize user-specific keys if needed (or handle in initialize_or_get_chain)
messages_key = f"messages_{user_id}"
if messages_key not in st.session_state:
    st.session_state[messages_key] = []


# --- Handle URL Query Parameter for Default Voice ---
if not st.session_state.session_initialized:
    try:
        query_params = st.query_params
        default_voice_param = query_params.get("voice")
        if default_voice_param and default_voice_param.lower() == "mufasa":
            st.session_state.selected_voice_name = MUFASA_VOICE_NAME
    except Exception as e:
        st.warning(f"Could not read query params: {e}")
    st.session_state.session_initialized = True

# --- Speech Synthesis (TTS) Setup ---
st.sidebar.header("Speech Output (TTS)")
tts_enabled = st.sidebar.toggle("Enable Speech Output", value=False, key="tts_enabled_toggle")

# TTS JavaScript Component to get voices
tts_component_value = components.html(
    f"""
    <script>
    // JS Code for populating voices (same as before)
    const synth = window.speechSynthesis;
    let voices = [];
    let safe_user_id = {json.dumps(user_id)};
    let lastSentVoiceDataString = sessionStorage.getItem('lastSentVoiceDataString_' + safe_user_id);
    // console.log("TTS Setup Component: Running"); // Reduce console noise

    function populateVoiceListAndSend() {{
        try {{
            voices = synth.getVoices();
            if (!voices || voices.length === 0) {{ return; }}
            // console.log('TTS Setup: Got', voices.length, 'voices.'); // Reduce noise
            voices.sort((a, b) => a.name.localeCompare(b.name));
            let mufasaLikeVoice = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') /* ...heuristic... */);
            const voiceOptions = voices.map(voice => ({{ name: voice.name, lang: voice.lang, default: voice.default }}));
            const customOptions = [
                {{ name: "{DEFAULT_VOICE_NAME}", lang: "", default: true }},
                {{ name: "{MUFASA_VOICE_NAME}", lang: mufasaLikeVoice ? mufasaLikeVoice.lang : "", default: false, internal_name: mufasaLikeVoice ? mufasaLikeVoice.name : null }}
            ];
            const newVoiceData = {{ voices: customOptions.concat(voiceOptions), type: "voices" }};
            const newVoiceDataString = JSON.stringify(newVoiceData.voices);
            if (newVoiceDataString !== lastSentVoiceDataString) {{
                 // console.log("TTS Setup: Sending updated voice list..."); // Reduce noise
                 lastSentVoiceDataString = newVoiceDataString;
                 sessionStorage.setItem('lastSentVoiceDataString_' + safe_user_id, newVoiceDataString);
                 if (window.Streamlit) {{ setTimeout(() => Streamlit.setComponentValue(newVoiceData), 0); }}
                 else {{ console.error("TTS Setup: Streamlit object not found."); }}
            }}
        }} catch (e) {{ console.error("TTS Setup: Error in populateVoiceListAndSend:", e); }}
    }}
    if (synth.onvoiceschanged !== undefined) {{ synth.onvoiceschanged = populateVoiceListAndSend; }}
    setTimeout(populateVoiceListAndSend, 200);
    </script>
    """,
    height=0
)

# Process the voice list returned from JS component
tts_voice_options = [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]
if isinstance(tts_component_value, dict) and tts_component_value.get("type") == "voices":
    st.session_state.tts_voices = tts_component_value.get("voices", [])
    custom_names = [v["name"] for v in st.session_state.tts_voices if v["name"] in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]]
    other_names = sorted([v["name"] for v in st.session_state.tts_voices if v["name"] not in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]])
    if custom_names or other_names: tts_voice_options = custom_names + other_names

# Ensure current selection is valid
current_selection = st.session_state.selected_voice_name
if current_selection not in tts_voice_options:
    current_selection = DEFAULT_VOICE_NAME
    st.session_state.selected_voice_name = current_selection

# Get user's selection from selectbox
selected_voice_name_from_ui = st.sidebar.selectbox(
    "Select Voice:", options=tts_voice_options, key="voice_selector",
    index=tts_voice_options.index(current_selection)
)
if selected_voice_name_from_ui != st.session_state.selected_voice_name:
    st.session_state.selected_voice_name = selected_voice_name_from_ui
    st.rerun()

# --- Display Mufasa Image Conditionally ---
if st.session_state.selected_voice_name == MUFASA_VOICE_NAME:
    st.sidebar.image(MUFASA_IMAGE_URL, width=150, caption="Mufasa Mode")

# --- Speech Recognition (STT) Setup ---
st.sidebar.header("Speech Input (STT)")
st.sidebar.caption("Click mic to toggle recording. (Browser support varies)")

# Toggle listening state on button press
mic_pressed = st.sidebar.button("🎤 Toggle Recording", key="stt_button")
if mic_pressed:
    st.session_state.stt_listening_toggle = not st.session_state.stt_listening_toggle
    if st.session_state.stt_listening_toggle: st.session_state.last_stt_processed = None
    st.rerun() # Rerun to update the JS component

# STT JavaScript Component
stt_component_value = components.html(
    f"""
    <script>
         // STT JavaScript code (same as before)
         const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
         const statusDiv = document.getElementById('stt-status');
         const shouldBeListening = {str(st.session_state.stt_listening_toggle).lower()};
         let recognition = window.stt_recognition;
         window.stt_listening = window.stt_listening || false;
         // console.log("STT Component: Running. Should listen:", shouldBeListening, "Currently listening:", window.stt_listening); // Reduce noise

         function sendValue(value) {{ /* ... */ if (window.Streamlit) Streamlit.setComponentValue({{ text: value, type: "stt_result" }}); }}

         if (SpeechRecognition) {{
             if (!recognition) {{
                 // console.log("STT: Initializing SpeechRecognition."); // Reduce noise
                 recognition = new SpeechRecognition();
                 window.stt_recognition = recognition;
                 recognition.continuous = false; recognition.interimResults = false;
                 recognition.onstart = () => {{ window.stt_listening = true; if(statusDiv) statusDiv.textContent = 'Listening...'; }};
                 recognition.onresult = (event) => {{ sendValue(event.results[0][0].transcript); }};
                 recognition.onerror = (event) => {{ window.stt_listening = false; if(statusDiv) statusDiv.textContent = `Error: ${{event.error}}`; console.error("STT Error:", event.error); }};
                 recognition.onend = () => {{ window.stt_listening = false; if(statusDiv) statusDiv.textContent = 'Mic idle.'; }};
             }}

             if (shouldBeListening && !window.stt_listening) {{
                 // console.log("STT: Attempting to start..."); // Reduce noise
                 try {{ recognition.start(); }} catch (e) {{ if (e.name !== 'InvalidStateError') {{ if(statusDiv) statusDiv.textContent = `Start Error: ${{e.message}}`; }} }}
             }} else if (!shouldBeListening && window.stt_listening) {{
                 // console.log("STT: Attempting to stop..."); // Reduce noise
                 try {{ recognition.stop(); }} catch(e) {{ /* Ignore */ }}
             }}
             if (statusDiv) {{ statusDiv.textContent = shouldBeListening ? (window.stt_listening ? 'Listening...' : 'Mic Starting...') : 'Mic idle.'; }}

         }} else {{ if(statusDiv) statusDiv.textContent = 'Speech Rec not supported.'; }}
    </script>
    <div id="stt-status">Mic status...</div>
    """,
    height=50, scrolling=False
)

# Check if STT component returned a result
recognized_text = ""
if isinstance(stt_component_value, dict) and stt_component_value.get("type") == "stt_result":
    new_text = stt_component_value.get("text", "")
    if new_text and new_text != st.session_state.last_stt_processed:
         # st.write(f"STT Result Received: {new_text}") # Debug print
         recognized_text = new_text
         st.session_state.last_stt_processed = new_text
         st.session_state.stt_output = recognized_text
         st.session_state.stt_listening_toggle = False # Turn off listening toggle
         st.rerun() # Rerun to process the input

# --- LLM Model Selection ---
# AVAILABLE_GROQ_MODELS already defined globally

# Initialize state variable for LLM model selection if needed
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = AVAILABLE_GROQ_MODELS[0]

# LLM Model selection dropdown
selected_llm_model_name = st.sidebar.selectbox(
    "Select LLM Model:", options=AVAILABLE_GROQ_MODELS, key="llm_model_selector",
    index=AVAILABLE_GROQ_MODELS.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in AVAILABLE_GROQ_MODELS else 0
)
# Update session state if the LLM selection changes
if selected_llm_model_name != st.session_state.selected_model_name:
    st.session_state.selected_model_name = selected_llm_model_name
    # Rerun needed because the chain/memory depends on the model name
    st.rerun()
st.sidebar.markdown(f"**Current LLM:** `{st.session_state.selected_model_name}`")

# --- Load Groq API Key ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key: st.error("Groq API Key not found."); st.stop()

# --- Initialize LLM, Memory, and Chain ---
def initialize_or_get_chain_mcp(model_name, user_id):
    """Initializes or retrieves the conversation chain and memory."""
    chain_key = f"conversation_chain_{user_id}_{model_name}"
    memory_key = f"memory_{user_id}_{model_name}"
    messages_key = f"messages_{user_id}"
    current_model_key = f"current_model_for_user_{user_id}"

    # Re-initialize chain if model selection changed for this user
    if st.session_state.get(current_model_key) != model_name:
        keys_to_delete = [k for k in st.session_state if k.startswith(f"conversation_chain_{user_id}_") or k.startswith(f"memory_{user_id}_")]
        for k in keys_to_delete:
             if k in st.session_state: del st.session_state[k]
        st.session_state[current_model_key] = model_name

    if chain_key not in st.session_state:
        try: llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        except Exception as e: st.error(f"Groq init error for '{model_name}': {e}"); st.stop()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        loaded_mcp_events = load_mcp_history_from_mongo(user_id)
        ui_messages = []
        for event in loaded_mcp_events:
            role, content = event.get("role"), event.get("content", "")
            if role == "user": memory.chat_memory.add_message(HumanMessage(content=content))
            elif role == "assistant": memory.chat_memory.add_message(AIMessage(content=content))
            if role in ["user", "assistant"]: ui_messages.append({"role": role, "content": content})

        st.session_state[memory_key] = memory
        st.session_state[messages_key] = ui_messages

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"Assistant on {model_name}. User ID: {user_id}. Be helpful."),
            MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")
        ])
        chain = ConversationChain(llm=llm, prompt=prompt_template, memory=st.session_state[memory_key], verbose=False)
        st.session_state[chain_key] = chain
        st.session_state[current_model_key] = model_name # Track model used for this chain

    return st.session_state[chain_key]

# Get the chain based on the *currently selected* LLM model name in session state
conversation_chain = initialize_or_get_chain_mcp(st.session_state.selected_model_name, user_id)
messages_key = f"messages_{user_id}"
if messages_key not in st.session_state: st.session_state[messages_key] = []

# --- Display Chat History ---
latest_assistant_response = ""
if messages_key in st.session_state:
    for message in st.session_state[messages_key]:
        role, content = message.get("role"), message.get("content")
        if role in ["user", "assistant"] and content:
             with st.chat_message(role): st.markdown(content)
             if role == "assistant": latest_assistant_response = content

# --- Handle Text Input OR Speech Input ---
user_input_text = st.chat_input("Ask something (or use mic):")
user_input_stt = st.session_state.get("stt_output", "")

final_user_input = None
input_source = None

if user_input_text:
    final_user_input = user_input_text; input_source = "text"
    st.session_state.stt_output = ""; st.session_state.last_stt_processed = None
elif user_input_stt:
    final_user_input = user_input_stt; input_source = "stt"
    st.session_state.stt_output = "" # Clear state immediately

# Process input if received from either source
if final_user_input:
    # st.write(f"Processing input ({input_source}): {final_user_input}") # Debug print
    # Add user message to display state and save to DB
    if not st.session_state[messages_key] or st.session_state[messages_key][-1].get("content") != final_user_input or st.session_state[messages_key][-1].get("role") != "user":
        st.session_state[messages_key].append({"role": "user", "content": final_user_input})
        # Rerun will handle display update if input came from STT
        if input_source == "text":
             with st.chat_message("user"): st.markdown(final_user_input)

    user_mcp_event = create_mcp_event(user_id, "user", final_user_input)
    save_mcp_event_to_mongo(user_mcp_event)

    # Get response from LLM
    with st.spinner(f"Thinking... ({st.session_state.selected_model_name})"):
        try:
            response = conversation_chain.predict(input=final_user_input)
            # st.write(f"LLM Response: {response}") # Debug print

            st.session_state[messages_key].append({"role": "assistant", "content": response})
            assistant_mcp_event = create_mcp_event(user_id, "assistant", response, model_name=st.session_state.selected_model_name)
            save_mcp_event_to_mongo(assistant_mcp_event)

            st.session_state.text_to_speak_trigger = response
            st.session_state.last_spoken_content = response

            st.rerun() # Rerun to update display and trigger TTS

        except Exception as e:
            st.error(f"An error occurred while getting response: {e}")


# --- Trigger TTS Component ---
text_to_speak = st.session_state.get("text_to_speak_trigger")
selected_voice = st.session_state.selected_voice_name # Use current selection

if tts_enabled and text_to_speak:
    st.session_state.text_to_speak_trigger = None # Clear trigger BEFORE calling component

    components.html(
        f"""
        <script>
            // TTS Trigger JS (with robust voice loading)
            const textToSpeak = {json.dumps(text_for_js)}; // Renamed variable for clarity
            const voiceNameToUse = {json.dumps(voice_for_js)}; // Renamed variable for clarity

            function speak(text, voiceName) {{
                const synth = window.speechSynthesis;
                const isTTSEnabled = {str(tts_enabled).lower()};
                // console.log("TTS Trigger: Trying to speak:", text, "Voice:", voiceName, "Enabled:", isTTSEnabled); // Reduce noise
                if (!isTTSEnabled || !text) return;

                let voices = synth.getVoices();
                if (!voices || voices.length === 0) {{
                    console.warn("TTS Trigger: Voices not ready, retrying...");
                    setTimeout(() => {{
                        voices = synth.getVoices();
                        if (!voices || voices.length === 0) {{ console.error("TTS Trigger: Voices still not ready."); return; }}
                        // console.log("TTS Trigger: Voices loaded on retry."); // Reduce noise
                        _executeSpeak(text, voiceName, voices, synth);
                    }}, 300);
                    return;
                }}
                 _executeSpeak(text, voiceName, voices, synth);
            }}

            function _executeSpeak(text, voiceName, voices, synth) {{
                 if (synth.speaking) {{ synth.cancel(); }} // Cancel previous speech
                 const utterThis = new SpeechSynthesisUtterance(text);
                 utterThis.onerror = (event) => console.error("TTS Trigger Error:", event);
                 // utterThis.onend = () => console.log("TTS Trigger: Speech finished."); // Reduce noise

                 let voiceToUse = null;
                 // Find voice logic... (ensure voices array is valid)
                 if(voices && voices.length > 0) {{
                     if (voiceName === "{DEFAULT_VOICE_NAME}") {{ voiceToUse = voices.find(v => v.default) || voices[0]; }}
                     else if (voiceName === "{MUFASA_VOICE_NAME}") {{ /* ...heuristic... */ voiceToUse = voices.find(...) || voices.find(v => v.default) || voices[0]; }}
                     else {{ voiceToUse = voices.find(v => v.name === voiceName); }}
                 }}

                 if (voiceToUse) {{ utterThis.voice = voiceToUse; /* console.log("TTS Trigger: Using voice:", voiceToUse.name); */ }}
                 else {{ console.warn(`TTS Trigger: Voice '${{voiceName}}' not found.`); }}

                 setTimeout(() => {{ /* console.log("TTS Trigger: Speaking..."); */ synth.speak(utterThis); }}, 100); // Delay speak slightly
            }}

            if (textToSpeak) {{ speak(textToSpeak, voiceNameToUse); }}
            // else {{ console.log("TTS Trigger: No text to speak."); }} // Reduce noise
        </script>
        """, height=0 )


# --- Placeholder for Gmail ---
# st.sidebar.button("Connect Gmail (Placeholder)")