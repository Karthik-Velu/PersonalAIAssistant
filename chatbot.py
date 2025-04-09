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
        client = MongoClient(uri, server_api=ServerApi('1'), connectTimeoutMS=5000) # Added timeout
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ping')
        # st.success("MongoDB connected!") # Reduce verbose output
        return client
    except pymongo.errors.ConfigurationError as e:
         st.error(f"MongoDB Configuration Error: {e}. Check connection string and credentials format.")
         st.stop()
    except pymongo.errors.OperationFailure as e:
         st.error(f"MongoDB Authentication Error: {e}. Check database username/password.")
         st.stop()
    except Exception as e: # Catch other potential errors during init
        st.error(f"Failed to initialize MongoDB client: {e}")
        st.stop()

mongo_client = initialize_mongo_client()
DB_NAME = st.secrets.get("MONGO_DB_NAME", "chatbot_db")
COLLECTION_NAME = st.secrets.get("MONGO_COLLECTION_NAME", "mcp_context_events")
try:
    db = mongo_client[DB_NAME]
    mcp_collection = db[COLLECTION_NAME]
    # Ensure index exists for efficient querying
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
     # Clear potentially cached data for the previous user if needed
     # (e.g., clear messages, reset chain - depends on desired behavior)
     st.rerun() # Rerun to reload data for the new user

# Ensure user_id is not empty before proceeding
user_id = st.session_state.user_id
if not user_id:
    st.warning("Please enter a User ID to begin.")
    st.stop()
st.sidebar.markdown(f"**User ID:** `{user_id}`")


# --- MCP Structure & MongoDB Functions ---
def create_mcp_event(user_id, role, content, model_name=None, metadata=None):
    """Creates a dictionary representing an MCP-like event."""
    event = {
        "user_id": user_id,
        "event_id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "metadata": metadata or {},
        "tool_calls": None, # Placeholder
        "tool_results": None, # Placeholder
    }
    if model_name and role == "assistant":
        event["metadata"]["model_used"] = model_name
    return event

def save_mcp_event_to_mongo(mcp_event):
    """Saves an MCP event document to MongoDB."""
    if not mcp_event.get("user_id"):
        st.warning("Attempted to save event without User ID.")
        return
    try:
        mcp_collection.insert_one(mcp_event)
    except Exception as e:
        st.error(f"Failed to save event to MongoDB: {e}") # Log error but don't stop app

def load_mcp_history_from_mongo(user_id, limit=50):
    """Loads the last 'limit' MCP events from MongoDB for a user."""
    if not user_id: return []
    try:
        cursor = mcp_collection.find({"user_id": user_id}).sort("timestamp", pymongo.DESCENDING).limit(limit)
        # Convert MongoDB docs (which might have _id) to plain dicts if needed, ensure timestamp is handled
        # For now, assume to_dict() or direct use is fine if _id is not an issue downstream
        messages_db = list(cursor)
        return messages_db[::-1] # Reverse for chronological order
    except Exception as e:
        st.error(f"Failed to load history from MongoDB: {e}")
        return []


# --- Handle URL Query Parameter for Default Voice ---
# Initialize session state keys if they don't exist
if "session_initialized" not in st.session_state: st.session_state.session_initialized = False
if "selected_voice_name" not in st.session_state: st.session_state.selected_voice_name = DEFAULT_VOICE_NAME

if not st.session_state.session_initialized:
    try:
        query_params = st.query_params
        default_voice_param = query_params.get("voice")
        if default_voice_param and default_voice_param.lower() == "mufasa":
            st.session_state.selected_voice_name = MUFASA_VOICE_NAME
        # No 'else' needed here, default is already set if key didn't exist
    except Exception as e:
        st.warning(f"Could not read query params on initial load: {e}")
        # Default is already set if key didn't exist
    st.session_state.session_initialized = True


# --- Speech Synthesis (TTS) Setup ---
st.sidebar.header("Speech Output (TTS)")
tts_enabled = st.sidebar.toggle("Enable Speech Output", value=False, key="tts_enabled_toggle")

if "tts_voices" not in st.session_state: st.session_state.tts_voices = []
# selected_voice_name already initialized above

# TTS JavaScript Component to get voices
# This component only sends data back, doesn't need key/default
tts_component_value = components.html(
    f"""
    <script>
    // JS Code for populating voices (same as before, with safety check)
    const synth = window.speechSynthesis;
    let voices = [];
    let safe_user_id = {json.dumps(user_id)};
    let lastSentVoiceDataString = sessionStorage.getItem('lastSentVoiceDataString_' + safe_user_id);

    function populateVoiceListAndSend() {{
        voices = synth.getVoices();
        if (!voices || voices.length === 0) {{ return; }} // Exit if no voices
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
             console.log("Sending updated voice list...");
             lastSentVoiceDataString = newVoiceDataString;
             sessionStorage.setItem('lastSentVoiceDataString_' + safe_user_id, newVoiceDataString);
             if (window.Streamlit) {{ setTimeout(() => Streamlit.setComponentValue(newVoiceData), 0); }}
             else {{ console.error("Streamlit object not found."); }}
        }}
    }}
    if (synth.onvoiceschanged !== undefined) {{ synth.onvoiceschanged = populateVoiceListAndSend; }}
    setTimeout(populateVoiceListAndSend, 150); // Slightly increased delay
    </script>
    """,
    height=0 # No visible element needed
)

# Process the voice list returned from JS component
tts_voice_options = [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME] # Start with defaults
# *** FIX: Added type check before calling .get() ***
if isinstance(tts_component_value, dict) and tts_component_value.get("type") == "voices":
    st.session_state.tts_voices = tts_component_value.get("voices", [])
    # Rebuild options list based on potentially updated voices
    custom_names = [v["name"] for v in st.session_state.tts_voices if v["name"] in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]]
    other_names = sorted([v["name"] for v in st.session_state.tts_voices if v["name"] not in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]])
    if custom_names or other_names:
        tts_voice_options = custom_names + other_names
    # else: keep the default list if JS fails to return valid data

# Ensure current selection is valid, default if not
current_selection = st.session_state.selected_voice_name
if current_selection not in tts_voice_options:
    current_selection = DEFAULT_VOICE_NAME
    st.session_state.selected_voice_name = current_selection

# Get user's selection from selectbox
selected_voice_name_from_ui = st.sidebar.selectbox(
    "Select Voice:", options=tts_voice_options, key="voice_selector",
    index=tts_voice_options.index(current_selection) # Set default based on current state
)
# Update session state ONLY if the selection changes
if selected_voice_name_from_ui != st.session_state.selected_voice_name:
    st.session_state.selected_voice_name = selected_voice_name_from_ui
    st.rerun() # Rerun if voice selection changes to update image etc.

# --- Display Mufasa Image Conditionally ---
if st.session_state.selected_voice_name == MUFASA_VOICE_NAME:
    st.sidebar.image(MUFASA_IMAGE_URL, width=150, caption="Mufasa Mode")

# --- Speech Recognition (STT) Setup ---
st.sidebar.header("Speech Input (STT)")
st.sidebar.caption("Click mic to toggle recording. (Browser support varies)")
# Initialize states if they don't exist
if 'stt_output' not in st.session_state: st.session_state.stt_output = ""
if 'stt_listening_toggle' not in st.session_state: st.session_state.stt_listening_toggle = False
if 'last_stt_processed' not in st.session_state: st.session_state.last_stt_processed = None

# Toggle listening state on button press
mic_pressed = st.sidebar.button("🎤 Toggle Recording", key="stt_button")
if mic_pressed:
    st.session_state.stt_listening_toggle = not st.session_state.stt_listening_toggle
    # Clear last processed only if starting recording, not stopping
    if st.session_state.stt_listening_toggle:
        st.session_state.last_stt_processed = None

# STT JavaScript Component
# This component now receives the desired listening state via an f-string variable
# It sends back results via Streamlit.setComponentValue
stt_component_value = components.html(
    f"""
    <script>
         const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
         const statusDiv = document.getElementById('stt-status');
         // Get desired state from Python variable embedded in the f-string
         const shouldBeListening = {str(st.session_state.stt_listening_toggle).lower()};
         let recognition = null;
         // Keep track of JS internal listening state separately
         window.stt_listening = window.stt_listening || false;

         function sendValue(value) {{
             if (window.Streamlit) {{
                 Streamlit.setComponentValue({{ text: value, type: "stt_result" }});
             }} else {{ console.error("Streamlit object not found."); }}
         }}

         if (SpeechRecognition) {{
             // Initialize recognition only once or if it doesn't exist
             if (!window.stt_recognition) {{
                 window.stt_recognition = new SpeechRecognition();
                 recognition = window.stt_recognition;
                 recognition.continuous = false; recognition.interimResults = false;
                 recognition.onstart = () => {{ window.stt_listening = true; if(statusDiv) statusDiv.textContent = 'Listening...'; }};
                 recognition.onresult = (event) => {{ sendValue(event.results[0][0].transcript); }}; // Send only final transcript
                 recognition.onerror = (event) => {{ window.stt_listening = false; if(statusDiv) statusDiv.textContent = `Error: ${{event.error}}`; }};
                 recognition.onend = () => {{ window.stt_listening = false; if(statusDiv) statusDiv.textContent = 'Mic idle.'; }};
             }}
             recognition = window.stt_recognition; // Use the stored instance

             // Sync state: Start/Stop based on shouldBeListening vs internal state
             if (shouldBeListening && !window.stt_listening) {{
                 try {{
                     recognition.start();
                 }} catch (e) {{
                     // Ignore errors if already started (can happen on fast reruns)
                     if (e.name !== 'InvalidStateError') {{
                         if(statusDiv) statusDiv.textContent = `Start Error: ${{e.message}}`;
                         console.error("STT start error:", e);
                     }}
                 }}
             }} else if (!shouldBeListening && window.stt_listening) {{
                 recognition.stop();
             }}
             // Update status div based on Python state passed in
             if (statusDiv) {{
                 statusDiv.textContent = shouldBeListening ? (window.stt_listening ? 'Listening...' : 'Starting...') : 'Mic idle.';
             }}

         }} else {{
              if(statusDiv) statusDiv.textContent = 'Speech Recognition not supported.';
         }}
    </script>
    <div id="stt-status">Mic status...</div>
    """,
    height=50, scrolling=False
)

# Check if STT component returned a result
recognized_text = ""
# Check type before using .get()
if isinstance(stt_component_value, dict) and stt_component_value.get("type") == "stt_result":
    new_text = stt_component_value.get("text", "")
    if new_text and new_text != st.session_state.last_stt_processed:
         recognized_text = new_text
         st.session_state.last_stt_processed = new_text # Mark as processed
         st.session_state.stt_output = recognized_text # Store for input handling
         st.session_state.stt_listening_toggle = False # Turn off listening after getting result
         st.rerun() # Rerun to process the input

# --- Model Selection & Groq API Key ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key: st.error("Groq API Key not found."); st.stop()

# --- Initialize LLM, Memory, and Chain ---
# (Ensure this uses the selected model name from session state)
def initialize_or_get_chain_mcp(model_name, user_id):
    chain_key = f"conversation_chain_{user_id}_{model_name}"
    memory_key = f"memory_{user_id}_{model_name}"
    messages_key = f"messages_{user_id}" # UI messages tied to user

    # Reset if model changed
    current_model_key = f"current_model_for_user_{user_id}"
    if st.session_state.get(current_model_key) != model_name:
        if chain_key in st.session_state: del st.session_state[chain_key]
        if memory_key in st.session_state: del st.session_state[memory_key]
        st.session_state[current_model_key] = model_name
        # Clear UI messages when model changes to avoid confusion? Optional.
        # if messages_key in st.session_state: del st.session_state[messages_key]

    # Initialize if not present in session state
    if chain_key not in st.session_state:
        # st.info(f"Initializing chain for model: {model_name}") # Less verbose
        try: llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        except Exception as e: st.error(f"Groq init error: {e}"); st.stop()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        loaded_mcp_events = load_mcp_history_from_mongo(user_id) # Load from DB

        ui_messages = [] # Build UI messages from loaded history
        for event in loaded_mcp_events:
            role, content = event.get("role"), event.get("content", "")
            # Populate LangChain memory
            if role == "user": memory.chat_memory.add_message(HumanMessage(content=content))
            elif role == "assistant": memory.chat_memory.add_message(AIMessage(content=content))
            # Populate UI list
            if role in ["user", "assistant"]: ui_messages.append({"role": role, "content": content})

        st.session_state[memory_key] = memory
        st.session_state[messages_key] = ui_messages # Set UI messages based on loaded history

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"Assistant on {model_name}. User ID: {user_id}."),
            MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")
        ])
        chain = ConversationChain(llm=llm, prompt=prompt_template, memory=st.session_state[memory_key], verbose=False)
        st.session_state[chain_key] = chain

    return st.session_state[chain_key]

# Use the selected voice name from session state
conversation_chain = initialize_or_get_chain_mcp(st.session_state.selected_model_name, user_id)
messages_key = f"messages_{user_id}" # Key for UI messages

# Initialize UI message list in session state if it wasn't populated by loading history
if messages_key not in st.session_state:
     st.session_state[messages_key] = []

# --- Display Chat History ---
latest_assistant_response = "" # Track latest response for TTS trigger
# Ensure message list exists before iterating
if messages_key in st.session_state:
    for message in st.session_state[messages_key]:
        role = message.get("role")
        content = message.get("content")
        if role in ["user", "assistant"] and content: # Check role and content
             with st.chat_message(role): st.markdown(content)
             if role == "assistant": latest_assistant_response = content

# --- Handle Text Input OR Speech Input ---
user_input_text = st.chat_input("Ask something (or use mic):")
user_input_stt = st.session_state.get("stt_output", "") # Get STT result from state

final_user_input = None
input_source = None

if user_input_text:
    final_user_input = user_input_text
    input_source = "text"
    st.session_state.stt_output = "" # Clear STT if text is used
    st.session_state.last_stt_processed = None
elif user_input_stt:
    final_user_input = user_input_stt
    input_source = "stt"
    st.session_state.stt_output = "" # Clear state immediately after reading

# Process input if received from either source
if final_user_input:
    # Add user message to display state and save to DB
    # Check if message already exists (can happen with STT reruns)
    if not st.session_state[messages_key] or st.session_state[messages_key][-1].get("content") != final_user_input or st.session_state[messages_key][-1].get("role") != "user":
        st.session_state[messages_key].append({"role": "user", "content": final_user_input})
        # Display message manually only if it came from text input
        if input_source == "text":
             with st.chat_message("user"): st.markdown(final_user_input)

    # Save user event to DB
    user_mcp_event = create_mcp_event(user_id, "user", final_user_input)
    save_mcp_event_to_mongo(user_mcp_event)

    # Get response from LLM
    with st.spinner(f"Thinking... ({st.session_state.selected_model_name})"):
        try:
            response = conversation_chain.predict(input=final_user_input)

            # Add assistant response to display state and save to DB
            st.session_state[messages_key].append({"role": "assistant", "content": response})
            assistant_mcp_event = create_mcp_event(user_id, "assistant", response, model_name=st.session_state.selected_model_name)
            save_mcp_event_to_mongo(assistant_mcp_event)

            # Update latest response for TTS trigger
            latest_assistant_response = response

            # Rerun to update the chat display and trigger TTS component
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred while getting response: {e}")
            # Consider removing the user message from display state if LLM fails
            # if st.session_state[messages_key] and st.session_state[messages_key][-1].get("role") == "user":
            #    st.session_state[messages_key].pop()


# --- Trigger TTS Component ---
# Use a separate state variable to track text intended for speech
if "text_to_speak_trigger" not in st.session_state:
    st.session_state.text_to_speak_trigger = None

# If there's a new assistant response, set it as the trigger text
if latest_assistant_response and latest_assistant_response != st.session_state.get("last_spoken_content", ""):
    st.session_state.text_to_speak_trigger = latest_assistant_response
    st.session_state.last_spoken_content = latest_assistant_response # Remember what was last generated

# Trigger the component only if TTS is enabled and there's text to speak
if tts_enabled and st.session_state.text_to_speak_trigger:
    text_for_js = st.session_state.text_to_speak_trigger
    voice_for_js = st.session_state.selected_voice_name
    # Clear the trigger immediately after reading it
    st.session_state.text_to_speak_trigger = None

    components.html(
        f"""
        <script>
            // This script instance is only for triggering speech
            const textToSpeak = {json.dumps(text_for_js)};
            const voiceNameToUse = {json.dumps(voice_for_js)};

            function speak(text, voiceName) {{
                const synth = window.speechSynthesis;
                const isTTSEnabled = {str(tts_enabled).lower()};
                if (!isTTSEnabled || !text) return;

                let voices = synth.getVoices();
                if (!voices || voices.length === 0) {{
                    console.warn("TTS voices not ready for speaking.");
                    // Attempt to load them again? Might be needed in edge cases.
                    // setTimeout(() => speak(text, voiceName), 200);
                    return;
                }}

                // Cancel anything currently speaking before starting new utterance
                if (synth.speaking) {{ synth.cancel(); }}

                const utterThis = new SpeechSynthesisUtterance(text);
                utterThis.onerror = (event) => console.error("TTS Error:", event);

                let voiceToUse = null;
                // Find voice logic (ensure 'voices' is populated)
                if (voiceName === "{DEFAULT_VOICE_NAME}") {{
                    voiceToUse = voices.find(v => v.default) || voices[0];
                }} else if (voiceName === "{MUFASA_VOICE_NAME}") {{
                     let mufasaInternalName = null;
                     let mufasaOption = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') /* ...heuristic... */ );
                     if (mufasaOption) mufasaInternalName = mufasaOption.name;
                     if (mufasaInternalName) voiceToUse = voices.find(v => v.name === mufasaInternalName);
                     if (!voiceToUse) voiceToUse = voices.find(v => v.default) || voices[0]; // Fallback
                }}
                else {{
                    voiceToUse = voices.find(v => v.name === voiceName);
                }}

                if (voiceToUse) {{
                    utterThis.voice = voiceToUse;
                    console.log("Using voice:", voiceToUse.name);
                }} else {{
                    console.warn(`Voice '${{voiceName}}' not found. Using default.`);
                }}
                // Small delay helps prevent issues on some browsers
                setTimeout(() => synth.speak(utterThis), 50);
            }}

            // Call speak immediately with the passed text and voice
            if (textToSpeak) {{
                 speak(textToSpeak, voiceNameToUse);
            }}
        </script>
        """, height=0 )


# --- Placeholder for Gmail ---
# st.sidebar.button("Connect Gmail (Placeholder)")