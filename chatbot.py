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

# --- MongoDB Initialization --- (Same as before)
@st.cache_resource
def initialize_mongo_client():
    """Initializes and returns the MongoDB client."""
    try:
        mongo_user = st.secrets.get("MONGO_USER")
        mongo_password = st.secrets.get("MONGO_PASSWORD")
        mongo_cluster_uri = st.secrets.get("MONGO_CLUSTER_URI")
        if not all([mongo_user, mongo_password, mongo_cluster_uri]):
            st.error("MongoDB credentials not found.")
            st.stop()
        encoded_user = urllib.parse.quote_plus(mongo_user)
        encoded_password = urllib.parse.quote_plus(mongo_password)
        uri = f"mongodb+srv://{encoded_user}:{encoded_password}@{mongo_cluster_uri}/?retryWrites=true&w=majority"
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Failed to initialize MongoDB client: {e}")
        st.stop()

mongo_client = initialize_mongo_client()
DB_NAME = st.secrets.get("MONGO_DB_NAME", "chatbot_db")
COLLECTION_NAME = st.secrets.get("MONGO_COLLECTION_NAME", "mcp_context_events")
try:
    db = mongo_client[DB_NAME]
    mcp_collection = db[COLLECTION_NAME]
    mcp_collection.create_index([("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
except Exception as e:
    st.error(f"Failed to access MongoDB or create index: {e}")
    st.stop()

# --- User Identification --- (Same as before)
st.sidebar.header("User Session")
user_id_input = st.sidebar.text_input("Enter a User ID:", key="user_id_input")
if user_id_input:
    st.session_state.user_id = user_id_input
else:
    if "user_id" not in st.session_state:
        st.warning("Please enter a User ID.")
        st.stop()
st.sidebar.markdown(f"**User ID:** `{st.session_state.user_id}`")
user_id = st.session_state.user_id

# --- MCP Structure & MongoDB Functions --- (Same as before)
def create_mcp_event(user_id, role, content, model_name=None, metadata=None):
    event = {
        "user_id": user_id, "event_id": str(uuid.uuid4()), "role": role,
        "content": content, "timestamp": datetime.datetime.now(datetime.timezone.utc),
        "metadata": metadata or {}, "tool_calls": None, "tool_results": None,
    }
    if model_name and role == "assistant": event["metadata"]["model_used"] = model_name
    return event

def save_mcp_event_to_mongo(mcp_event):
    if not mcp_event.get("user_id"): return
    try: mcp_collection.insert_one(mcp_event)
    except Exception as e: st.error(f"Failed to save event to MongoDB: {e}")

def load_mcp_history_from_mongo(user_id, limit=50):
    if not user_id: return []
    try:
        cursor = mcp_collection.find({"user_id": user_id}).sort("timestamp", pymongo.DESCENDING).limit(limit)
        return list(cursor)[::-1] # Reverse for chronological order
    except Exception as e: st.error(f"Failed to load history from MongoDB: {e}"); return []


# --- Handle URL Query Parameter for Default Voice ---
# This needs to run BEFORE the selectbox is rendered and AFTER session state is checked/initialized
if "session_initialized" not in st.session_state:
    try:
        # Check query params only on the first run of a session
        query_params = st.query_params
        default_voice_param = query_params.get("voice")
        if default_voice_param and default_voice_param.lower() == "mufasa":
            st.session_state.selected_voice_name = MUFASA_VOICE_NAME
        else:
            # Initialize with default if not set by query param
             if "selected_voice_name" not in st.session_state:
                 st.session_state.selected_voice_name = DEFAULT_VOICE_NAME
    except Exception as e: # Handle potential errors during query param access on first load
        st.warning(f"Could not read query params on initial load: {e}")
        if "selected_voice_name" not in st.session_state:
             st.session_state.selected_voice_name = DEFAULT_VOICE_NAME

    st.session_state.session_initialized = True # Mark session as initialized


# --- Speech Synthesis (TTS) Setup ---
st.sidebar.header("Speech Output (TTS)")
tts_enabled = st.sidebar.toggle("Enable Speech Output", value=False, key="tts_enabled_toggle")

# Placeholder for voices
if "tts_voices" not in st.session_state:
    st.session_state.tts_voices = []
# Ensure selected_voice_name is initialized if somehow missed above
if "selected_voice_name" not in st.session_state:
     st.session_state.selected_voice_name = DEFAULT_VOICE_NAME

# TTS JavaScript Component (same JS logic as before)
tts_component_value = components.html(
    f"""
    <script>
    const synth = window.speechSynthesis;
    let voices = [];
    let selectedVoice = null;
    let lastSpokenText = sessionStorage.getItem('lastSpokenText_{user_id}') || ""; // Use sessionStorage for persistence across reruns

    function populateVoiceListAndSend() {{
        voices = synth.getVoices().sort((a, b) => a.name.localeCompare(b.name));
        let mufasaLikeVoice = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') && !v.name.toLowerCase().includes('child') && !v.name.toLowerCase().includes('female') && (v.name.toLowerCase().includes('david') || v.name.toLowerCase().includes('mark') || v.name.toLowerCase().includes('james') || v.name.toLowerCase().includes('google') || v.name.toLowerCase().includes('microsoft david') || v.name.toLowerCase().includes('microsoft mark') || v.name.toLowerCase().includes('daniel')));
        const voiceOptions = voices.map(voice => ({{ name: voice.name, lang: voice.lang, default: voice.default }}));
        const customOptions = [
            {{ name: "{DEFAULT_VOICE_NAME}", lang: "", default: true }},
            {{ name: "{MUFASA_VOICE_NAME}", lang: mufasaLikeVoice ? mufasaLikeVoice.lang : "", default: false, internal_name: mufasaLikeVoice ? mufasaLikeVoice.name : null }}
        ];
        // Prevent sending if voices haven't changed significantly
        const currentSentVoices = JSON.stringify(Streamlit.componentValue?.voices);
        const newVoiceData = { voices: customOptions.concat(voiceOptions), type: "voices" };
        if (JSON.stringify(newVoiceData.voices) !== currentSentVoices) {{
             setTimeout(() => Streamlit.setComponentValue(newVoiceData), 0);
        }}
    }}

    if (synth.onvoiceschanged !== undefined) {{
        synth.onvoiceschanged = populateVoiceListAndSend;
    }}
    populateVoiceListAndSend(); // Initial call

    function speak(text, voiceName) {{
        // Use Python variable tts_enabled directly
        const isTTSEnabled = {str(tts_enabled).lower()};
        if (!isTTSEnabled || !text || text === lastSpokenText) return;
        if (synth.speaking) {{ synth.cancel(); }} // Cancel previous before speaking new

        lastSpokenText = text;
        sessionStorage.setItem('lastSpokenText_{user_id}', text); // Store in session storage

        const utterThis = new SpeechSynthesisUtterance(text);
        utterThis.onend = () => {{ lastSpokenText = ""; sessionStorage.removeItem('lastSpokenText_{user_id}'); }};
        utterThis.onerror = (event) => {{ console.error("TTS Error:", event); lastSpokenText = ""; sessionStorage.removeItem('lastSpokenText_{user_id}'); }};

        let voiceToUse = null;
        if (voiceName === "{DEFAULT_VOICE_NAME}") {{
             voiceToUse = voices.find(v => v.default) || voices[0];
        }} else if (voiceName === "{MUFASA_VOICE_NAME}") {{
             // Find internal name from the options potentially sent back earlier
             // This relies on the structure sent back to Streamlit
             // A more robust way might involve passing the internal name mapping if needed
             let mufasaInternalName = null;
             // Attempt to find the internal name (this part is heuristic)
             let mufasaOption = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') && !v.name.toLowerCase().includes('child') && !v.name.toLowerCase().includes('female') && (v.name.toLowerCase().includes('david') || v.name.toLowerCase().includes('mark') || v.name.toLowerCase().includes('james') || v.name.toLowerCase().includes('google') || v.name.toLowerCase().includes('microsoft david') || v.name.toLowerCase().includes('microsoft mark') || v.name.toLowerCase().includes('daniel')));
             if (mufasaOption) mufasaInternalName = mufasaOption.name;

             if (mufasaInternalName) {{
                 voiceToUse = voices.find(v => v.name === mufasaInternalName);
             }}
             if (!voiceToUse) voiceToUse = voices.find(v => v.default) || voices[0]; // Fallback
        }} else {{
            voiceToUse = voices.find(v => v.name === voiceName);
        }}

        if (voiceToUse) {{
             utterThis.voice = voiceToUse;
        }} else {{ console.warn(`Voice '${{voiceName}}' not found.`); }}

        setTimeout(() => synth.speak(utterThis), 50);
    }}

    // Component args are passed via the *second* components.html call below
    // This instance mainly handles voice list population
    </script>
    """,
    key="tts_js_setup_component",
    default={"type": "init"} # Pass initial default value
)


# Process the voice list returned from JS component
tts_voice_options = [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]
if tts_component_value and tts_component_value.get("type") == "voices":
    st.session_state.tts_voices = tts_component_value.get("voices", [])
    custom_names = [v["name"] for v in st.session_state.tts_voices if v["name"] in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]]
    other_names = sorted([v["name"] for v in st.session_state.tts_voices if v["name"] not in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]])
    tts_voice_options = custom_names + other_names

# Ensure the current selection is valid, default if not
current_selection = st.session_state.selected_voice_name
if current_selection not in tts_voice_options:
    current_selection = DEFAULT_VOICE_NAME
    st.session_state.selected_voice_name = current_selection

# Update selected voice based on user choice in selectbox
selected_voice_name = st.sidebar.selectbox(
    "Select Voice:",
    options=tts_voice_options,
    key="voice_selector",
    index=tts_voice_options.index(current_selection) # Use validated current selection
)
# Update session state ONLY if the selectbox value changes
if selected_voice_name != st.session_state.selected_voice_name:
    st.session_state.selected_voice_name = selected_voice_name
    # No rerun needed here, will be handled by Streamlit naturally


# --- Display Mufasa Image Conditionally ---
if st.session_state.selected_voice_name == MUFASA_VOICE_NAME:
    st.sidebar.image(MUFASA_IMAGE_URL, width=150, caption="Mufasa Mode")


# --- Speech Recognition (STT) Setup --- (Same JS logic as before)
st.sidebar.header("Speech Input (STT)")
st.sidebar.caption("Click mic, speak, click again. (Browser support varies)")
if 'stt_output' not in st.session_state: st.session_state.stt_output = ""
if 'stt_listening_toggle' not in st.session_state: st.session_state.stt_listening_toggle = False # Init toggle state

mic_pressed = st.sidebar.button("🎤 Start/Stop Recording", key="stt_button")
if mic_pressed:
    # Toggle the state when button is pressed
    st.session_state.stt_listening_toggle = not st.session_state.stt_listening_toggle

# Pass the toggle state to the JS component
stt_component_value = components.html(
    f"""
    <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const statusDiv = document.getElementById('stt-status');
        let recognition = null;
        let listening = false;
        let final_transcript = '';

        function sendValue(value) {{ Streamlit.setComponentValue({{ text: value, type: "stt_result" }}); }}

        if (!SpeechRecognition) {{ /* ... error handling ... */ }}
        else {{
            recognition = new SpeechRecognition();
            recognition.continuous = false; recognition.interimResults = false;
            recognition.onstart = () => {{ listening = true; final_transcript = ''; if(statusDiv) statusDiv.textContent = 'Listening...'; }};
            recognition.onresult = (event) => {{ final_transcript += event.results[0][0].transcript; sendValue(final_transcript); }};
            recognition.onerror = (event) => {{ listening = false; if(statusDiv) statusDiv.textContent = `Error: ${{event.error}}`; }};
            recognition.onend = () => {{ listening = false; if(statusDiv) statusDiv.textContent = 'Mic idle.'; }};
        }}

        function toggleListen() {{
            if (!recognition) return;
            if (listening) {{ recognition.stop(); }}
            else {{ try {{ recognition.start(); }} catch (e) {{ if(statusDiv) statusDiv.textContent = `Start Error: ${{e.message}}`; }} }}
        }}

         // Use componentArgs to receive the toggle signal from Python
         const args = Streamlit.componentArgs;
         // Check if the toggle signal exists and has changed since last time JS saw it
         if (args && typeof args.toggle_signal !== 'undefined' && args.toggle_signal !== window.lastToggleSignalState) {{
             window.lastToggleSignalState = args.toggle_signal; // Store the new state
             // Only call toggleListen if the signal indicates a change initiated by the button
             // This check prevents toggling on every script rerun
             if (args.triggered_by_button) {{
                  toggleListen();
             }}
         }}
    </script>
    <div id="stt-status">Mic idle. (Requires browser permission)</div>
    """,
    key="stt_js_component",
    # Pass the toggle state AND explicitly signal if button triggered it
    default={"type": "init", "toggle_signal": st.session_state.stt_listening_toggle, "triggered_by_button": False}, # Initial state
    # On button press, pass the NEW toggle state and mark triggered_by_button as True
    # This requires careful handling of reruns. A simpler way might be needed if this causes issues.
    # Let's try passing the state directly. JS needs to compare with its *previous* state.
    scrolling=False, height=50
)


# Check if STT component returned a result (logic improved slightly)
recognized_text = ""
if stt_component_value and stt_component_value.get("type") == "stt_result":
    new_text = stt_component_value.get("text", "")
    # Process only if it's a new, non-empty result different from last processed
    if new_text and new_text != st.session_state.get("last_stt_processed", ""):
         recognized_text = new_text
         st.session_state.last_stt_processed = new_text # Remember what we just processed
         st.session_state.stt_output = recognized_text # Update state for input handling
         st.rerun() # Rerun to process the new STT input immediately
    # If the value from component is empty or same as last, do nothing or clear state
    elif not new_text and "last_stt_processed" in st.session_state:
         del st.session_state["last_stt_processed"] # Clear if component sends empty


# --- Model Selection & Groq API Key --- (Same as before)
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key: st.error("Groq API Key not found."); st.stop()

# --- Initialize LLM, Memory, and Chain --- (Same as before)
def initialize_or_get_chain_mcp(model_name, user_id):
    chain_key = f"conversation_chain_{user_id}_{model_name}"
    memory_key = f"memory_{user_id}_{model_name}"
    messages_key = f"messages_{user_id}"

    if st.session_state.get("current_model_for_user_" + user_id) != model_name:
        if chain_key in st.session_state: del st.session_state[chain_key]
        if memory_key in st.session_state: del st.session_state[memory_key]
        st.session_state["current_model_for_user_" + user_id] = model_name

    if chain_key not in st.session_state:
        try: llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        except Exception as e: st.error(f"Groq init error: {e}"); st.stop()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        loaded_mcp_events = load_mcp_history_from_mongo(user_id)
        ui_messages = []
        for event in loaded_mcp_events:
            role, content = event.get("role"), event.get("content", "")
            if role == "user": memory.chat_memory.add_message(HumanMessage(content=content)); ui_messages.append({"role": "user", "content": content})
            elif role == "assistant": memory.chat_memory.add_message(AIMessage(content=content)); ui_messages.append({"role": "assistant", "content": content})

        st.session_state[memory_key] = memory
        st.session_state[messages_key] = ui_messages

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"Assistant on {model_name}. User ID: {user_id}."),
            MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")
        ])
        chain = ConversationChain(llm=llm, prompt=prompt_template, memory=st.session_state[memory_key], verbose=False)
        st.session_state[chain_key] = chain
    return st.session_state[chain_key]

conversation_chain = initialize_or_get_chain_mcp(selected_model_name, user_id)
messages_key = f"messages_{user_id}"
if messages_key not in st.session_state: st.session_state[messages_key] = []

# --- Display Chat History ---
latest_assistant_response = ""
for message in st.session_state[messages_key]:
    role, content = message.get("role"), message.get("content")
    if role and content:
         with st.chat_message(role): st.markdown(content)
         if role == "assistant": latest_assistant_response = content

# --- Handle Text Input OR Speech Input ---
# Use text input value directly if provided
user_input_text = st.chat_input("Ask something (or use mic):")
# Use STT result stored in session state if text input is empty
user_input_stt = st.session_state.get("stt_output", "")

final_user_input = None
input_source = None # Track where input came from

if user_input_text:
    final_user_input = user_input_text
    input_source = "text"
    st.session_state.stt_output = "" # Clear STT if text is used
    if "last_stt_processed" in st.session_state: del st.session_state["last_stt_processed"]
elif user_input_stt:
    final_user_input = user_input_stt
    input_source = "stt"
    # Clear the state variable immediately after reading it
    st.session_state.stt_output = ""
    # Keep last_stt_processed until next STT result comes


# If we have input (from either source)
if final_user_input:
    # Add user message to display state and save
    # Avoid duplicating message display if it came from STT and caused a rerun
    if input_source == "text":
        st.session_state[messages_key].append({"role": "user", "content": final_user_input})
        with st.chat_message("user"): st.markdown(final_user_input)
    elif input_source == "stt":
        # Check if message already exists from potential STT rerun display
        if not st.session_state[messages_key] or st.session_state[messages_key][-1].get("content") != final_user_input:
             st.session_state[messages_key].append({"role": "user", "content": final_user_input})
             # No need to display here, STT component causes rerun which displays history

    user_mcp_event = create_mcp_event(user_id, "user", final_user_input)
    save_mcp_event_to_mongo(user_mcp_event)

    # Get response
    with st.spinner(f"Thinking... ({selected_model_name})"):
        try:
            response = conversation_chain.predict(input=final_user_input)
            st.session_state[messages_key].append({"role": "assistant", "content": response})
            assistant_mcp_event = create_mcp_event(user_id, "assistant", response, model_name=selected_model_name)
            save_mcp_event_to_mongo(assistant_mcp_event)

            # Update latest response for TTS trigger
            latest_assistant_response = response

            # Display assistant response (will happen on next rerun if STT caused it)
            # For text input, display immediately is fine, but might double display
            # Let's rely on the history display loop at the top for consistency
            # with st.chat_message("assistant"): st.markdown(response) # Comment out immediate display

            # Rerun to update display cleanly, especially after STT
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")


# --- Trigger TTS Component ---
# Ensure this runs after potential reruns and state updates
if tts_enabled and latest_assistant_response and latest_assistant_response != st.session_state.get("last_spoken_trigger_text", ""):
    st.session_state.last_spoken_trigger_text = latest_assistant_response # Prevent re-triggering immediately
    components.html(
        f"""
        <script>
            // Simplified script assuming speak function is available globally from first component instance
            // In a real component, you'd manage scope better.
            const textToSpeak = {json.dumps(latest_assistant_response)};
            const voiceNameToUse = {json.dumps(st.session_state.selected_voice_name)};

            // Function definition might be needed if scope isn't shared
            function speak(text, voiceName) {{
                const synth = window.speechSynthesis;
                const isTTSEnabled = {str(tts_enabled).lower()};
                if (!isTTSEnabled || !text) return;
                if (synth.speaking) {{ synth.cancel(); }}

                const utterThis = new SpeechSynthesisUtterance(text);
                utterThis.onerror = (event) => console.error("TTS Error:", event);

                let voices = synth.getVoices(); // Get voices again
                let voiceToUse = null;
                // Simplified voice finding logic
                 if (voiceName === "{DEFAULT_VOICE_NAME}") voiceToUse = voices.find(v => v.default) || voices[0];
                 // Add Mufasa-like logic here if needed
                 else voiceToUse = voices.find(v => v.name === voiceName);

                 if (voiceToUse) utterThis.voice = voiceToUse;
                 else console.warn(`Voice '${{voiceName}}' not found for speaking.`);

                 setTimeout(() => synth.speak(utterThis), 50);
            }}

            if (textToSpeak) {{
                 speak(textToSpeak, voiceNameToUse);
            }}
        </script>
        """,
        height=0
    )
elif not latest_assistant_response and "last_spoken_trigger_text" in st.session_state:
     del st.session_state["last_spoken_trigger_text"] # Clear trigger state if no response


# <<< Placeholder for Gmail Integration >>>
# st.sidebar.button("Connect Gmail (Placeholder)")