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

# --- User Identification ---
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

# --- MCP Structure & MongoDB Functions ---
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
        return list(cursor)[::-1]
    except Exception as e: st.error(f"Failed to load history from MongoDB: {e}"); return []


# --- Handle URL Query Parameter for Default Voice ---
if "session_initialized" not in st.session_state:
    try:
        query_params = st.query_params
        default_voice_param = query_params.get("voice")
        if default_voice_param and default_voice_param.lower() == "mufasa":
            st.session_state.selected_voice_name = MUFASA_VOICE_NAME
        else:
             if "selected_voice_name" not in st.session_state:
                 st.session_state.selected_voice_name = DEFAULT_VOICE_NAME
    except Exception as e:
        st.warning(f"Could not read query params on initial load: {e}")
        if "selected_voice_name" not in st.session_state:
             st.session_state.selected_voice_name = DEFAULT_VOICE_NAME
    st.session_state.session_initialized = True


# --- Speech Synthesis (TTS) Setup ---
st.sidebar.header("Speech Output (TTS)")
tts_enabled = st.sidebar.toggle("Enable Speech Output", value=False, key="tts_enabled_toggle")

if "tts_voices" not in st.session_state: st.session_state.tts_voices = []
if "selected_voice_name" not in st.session_state: st.session_state.selected_voice_name = DEFAULT_VOICE_NAME

# TTS JavaScript Component (with fix for NameError)
tts_component_value = components.html(
    f"""
    <script>
    const synth = window.speechSynthesis;
    let voices = [];
    // Use sessionStorage to reduce re-sending voice list unnecessarily
    let lastSentVoiceDataString = sessionStorage.getItem('lastSentVoiceDataString_{user_id}');

    function populateVoiceListAndSend() {{
        voices = synth.getVoices(); // Get available voices

        // *** FIX START: Check if voices array is populated before proceeding ***
        if (!voices || voices.length === 0) {{
            console.log('Voices not ready yet or empty.');
            // Optionally try again after a short delay
            // setTimeout(populateVoiceListAndSend, 200);
            return; // Exit function if no voices loaded
        }}
        // *** FIX END ***

        // Sort voices by name
        voices.sort((a, b) => a.name.localeCompare(b.name));

        // Try to find a Mufasa-like voice (heuristic)
        let mufasaLikeVoice = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') && !v.name.toLowerCase().includes('child') && !v.name.toLowerCase().includes('female') && (v.name.toLowerCase().includes('david') || v.name.toLowerCase().includes('mark') || v.name.toLowerCase().includes('james') || v.name.toLowerCase().includes('google') || v.name.toLowerCase().includes('microsoft david') || v.name.toLowerCase().includes('microsoft mark') || v.name.toLowerCase().includes('daniel')));

        // Map voices to a simpler structure for sending to Streamlit
        const voiceOptions = voices.map(voice => ({{ name: voice.name, lang: voice.lang, default: voice.default }}));

        // Define custom options (Default and Mufasa-like)
        const customOptions = [
            {{ name: "{DEFAULT_VOICE_NAME}", lang: "", default: true }},
            {{ name: "{MUFASA_VOICE_NAME}", lang: mufasaLikeVoice ? mufasaLikeVoice.lang : "", default: false, internal_name: mufasaLikeVoice ? mufasaLikeVoice.name : null }}
        ];

        // Combine custom options and the rest of the voices
        const newVoiceData = {{ voices: customOptions.concat(voiceOptions), type: "voices" }};
        const newVoiceDataString = JSON.stringify(newVoiceData.voices);

        // Send voice options back to Streamlit ONLY if they have changed
        if (newVoiceDataString !== lastSentVoiceDataString) {{
             console.log("Sending updated voice list to Streamlit...");
             lastSentVoiceDataString = newVoiceDataString;
             sessionStorage.setItem('lastSentVoiceDataString_{user_id}', newVoiceDataString); // Store in session storage
             // Use a timeout to ensure Streamlit is ready
             setTimeout(() => Streamlit.setComponentValue(newVoiceData), 0);
        }} else {{
             // console.log("Voice list hasn't changed, not sending.");
        }}
    }}

    // Populate voices when they are loaded or changed
    if (synth.onvoiceschanged !== undefined) {{
        synth.onvoiceschanged = populateVoiceListAndSend;
    }}
    // Call it once initially after a small delay, in case voices are already loaded
    // but give onvoiceschanged a chance first.
    setTimeout(populateVoiceListAndSend, 100);


    // Speak function (remains mostly the same, ensure it uses the 'voices' array populated above)
    function speak(text, voiceName) {{
        const isTTSEnabled = {str(tts_enabled).lower()};
        // Ensure voices array is populated before trying to find voice by name
        if (!isTTSEnabled || !text || !voices || voices.length === 0) return;
        if (synth.speaking) {{ synth.cancel(); }}

        const utterThis = new SpeechSynthesisUtterance(text);
        utterThis.onerror = (event) => console.error("TTS Error:", event);

        let voiceToUse = null;
        if (voiceName === "{DEFAULT_VOICE_NAME}") {{
             voiceToUse = voices.find(v => v.default) || voices[0];
        }} else if (voiceName === "{MUFASA_VOICE_NAME}") {{
             // Heuristic find again (internal_name mapping might be lost across reruns)
             let mufasaInternalName = null;
             let mufasaOption = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male') /* ... rest of heuristic ... */ );
             if (mufasaOption) mufasaInternalName = mufasaOption.name;
             if (mufasaInternalName) voiceToUse = voices.find(v => v.name === mufasaInternalName);
             if (!voiceToUse) voiceToUse = voices.find(v => v.default) || voices[0];
        }} else {{
            voiceToUse = voices.find(v => v.name === voiceName);
        }}

        if (voiceToUse) {{ utterThis.voice = voiceToUse; }}
        else {{ console.warn(`Voice '${{voiceName}}' not found.`); }}

        setTimeout(() => synth.speak(utterThis), 50);
    }}

    // This component instance primarily handles voice list population.
    // The actual speaking is triggered by the second component call later.
    </script>
    """,
    key="tts_js_setup_component",
    default={"type": "init"}
)


# Process the voice list returned from JS component (remains same)
tts_voice_options = [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]
if tts_component_value and tts_component_value.get("type") == "voices":
    st.session_state.tts_voices = tts_component_value.get("voices", [])
    custom_names = [v["name"] for v in st.session_state.tts_voices if v["name"] in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]]
    other_names = sorted([v["name"] for v in st.session_state.tts_voices if v["name"] not in [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]])
    # Basic check to ensure we have some options
    if custom_names or other_names:
        tts_voice_options = custom_names + other_names
    else: # Fallback if JS fails to send voices
        tts_voice_options = [DEFAULT_VOICE_NAME, MUFASA_VOICE_NAME]


# Update selected voice based on user choice in selectbox (remains same)
current_selection = st.session_state.selected_voice_name
if current_selection not in tts_voice_options:
    current_selection = DEFAULT_VOICE_NAME
    st.session_state.selected_voice_name = current_selection
selected_voice_name = st.sidebar.selectbox(
    "Select Voice:", options=tts_voice_options, key="voice_selector",
    index=tts_voice_options.index(current_selection)
)
if selected_voice_name != st.session_state.selected_voice_name:
    st.session_state.selected_voice_name = selected_voice_name


# --- Display Mufasa Image Conditionally --- (Same as before)
if st.session_state.selected_voice_name == MUFASA_VOICE_NAME:
    st.sidebar.image(MUFASA_IMAGE_URL, width=150, caption="Mufasa Mode")


# --- Speech Recognition (STT) Setup --- (Same as before)
st.sidebar.header("Speech Input (STT)")
st.sidebar.caption("Click mic, speak, click again. (Browser support varies)")
if 'stt_output' not in st.session_state: st.session_state.stt_output = ""
if 'stt_listening_toggle' not in st.session_state: st.session_state.stt_listening_toggle = False
if 'last_stt_processed' not in st.session_state: st.session_state.last_stt_processed = None # Init state

mic_pressed = st.sidebar.button("🎤 Start/Stop Recording", key="stt_button")
if mic_pressed:
    st.session_state.stt_listening_toggle = not st.session_state.stt_listening_toggle
    # Reset last processed text when toggling to ensure new input is picked up
    st.session_state.last_stt_processed = None

stt_component_value = components.html(
    f"""
    <script>
        // STT JavaScript code (same as speech_chatbot_visual_cue_v1)
        // ... (Includes SpeechRecognition setup, onresult, onerror, onend, toggleListen) ...
        // Ensure toggleListen is correctly triggered based on state changes passed from Python
         const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
         const statusDiv = document.getElementById('stt-status');
         let recognition = null;
         let listening = false; // Internal JS listening state
         let final_transcript = '';

         function sendValue(value) {{ Streamlit.setComponentValue({{ text: value, type: "stt_result" }}); }}

         if (SpeechRecognition) {{
             recognition = new SpeechRecognition();
             recognition.continuous = false; recognition.interimResults = false;
             recognition.onstart = () => {{ listening = true; final_transcript = ''; if(statusDiv) statusDiv.textContent = 'Listening...'; }};
             recognition.onresult = (event) => {{ final_transcript += event.results[0][0].transcript; sendValue(final_transcript); }};
             recognition.onerror = (event) => {{ listening = false; if(statusDiv) statusDiv.textContent = `Error: ${{event.error}}`; }};
             recognition.onend = () => {{ listening = false; if(statusDiv) statusDiv.textContent = 'Mic idle.'; }};

             // Function to sync listening state and start/stop recognition
             function syncAndToggleListen(shouldBeListening) {{
                 if (!recognition) return;
                 if (shouldBeListening && !listening) {{
                     try {{ recognition.start(); }} catch (e) {{ if(statusDiv) statusDiv.textContent = `Start Error: ${{e.message}}`; }}
                 }} else if (!shouldBeListening && listening) {{
                     recognition.stop();
                 }}
             }}

             // Get the desired state from Streamlit args
             const args = Streamlit.componentArgs;
             if (args && typeof args.should_listen !== 'undefined') {{
                 // Sync JS state with Python state passed via args
                 syncAndToggleListen(args.should_listen);
             }}
         }} else {{
              if(statusDiv) statusDiv.textContent = 'Speech Recognition not supported.';
         }}
    </script>
    <div id="stt-status">Mic idle. (Requires browser permission)</div>
    """,
    key="stt_js_component",
    # Pass the desired listening state from Python session state
    default={"type": "init", "should_listen": st.session_state.stt_listening_toggle},
    scrolling=False, height=50
)

# Check if STT component returned a result (improved logic)
recognized_text = ""
if stt_component_value and stt_component_value.get("type") == "stt_result":
    new_text = stt_component_value.get("text", "")
    if new_text and new_text != st.session_state.last_stt_processed:
         recognized_text = new_text
         st.session_state.last_stt_processed = new_text
         st.session_state.stt_output = recognized_text
         # Set toggle state back to off after receiving result?
         st.session_state.stt_listening_toggle = False
         st.rerun()


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

# --- Display Chat History --- (Same as before)
latest_assistant_response = ""
for message in st.session_state[messages_key]:
    role, content = message.get("role"), message.get("content")
    if role and content:
         with st.chat_message(role): st.markdown(content)
         if role == "assistant": latest_assistant_response = content

# --- Handle Text Input OR Speech Input --- (Same logic as before)
user_input_text = st.chat_input("Ask something (or use mic):")
user_input_stt = st.session_state.get("stt_output", "")
final_user_input = None
input_source = None
if user_input_text:
    final_user_input = user_input_text; input_source = "text"
    st.session_state.stt_output = ""; st.session_state.last_stt_processed = None
elif user_input_stt:
    final_user_input = user_input_stt; input_source = "stt"
    st.session_state.stt_output = "" # Clear state after reading

if final_user_input:
    # Add user message to display state and save
    if input_source == "text" or not st.session_state[messages_key] or st.session_state[messages_key][-1].get("content") != final_user_input:
        st.session_state[messages_key].append({"role": "user", "content": final_user_input})
        # Only display manually if text input, STT relies on rerun history display
        if input_source == "text":
             with st.chat_message("user"): st.markdown(final_user_input)

    user_mcp_event = create_mcp_event(user_id, "user", final_user_input)
    save_mcp_event_to_mongo(user_mcp_event)

    # Get response
    with st.spinner(f"Thinking... ({selected_model_name})"):
        try:
            response = conversation_chain.predict(input=final_user_input)
            st.session_state[messages_key].append({"role": "assistant", "content": response})
            assistant_mcp_event = create_mcp_event(user_id, "assistant", response, model_name=selected_model_name)
            save_mcp_event_to_mongo(assistant_mcp_event)
            latest_assistant_response = response # Update for TTS
            st.rerun() # Rerun to display new messages and trigger TTS component
        except Exception as e:
            st.error(f"An error occurred: {e}")


# --- Trigger TTS Component --- (Same as before, relies on latest_assistant_response)
if tts_enabled and latest_assistant_response and latest_assistant_response != st.session_state.get("last_spoken_trigger_text", ""):
    st.session_state.last_spoken_trigger_text = latest_assistant_response
    components.html(
        f"""
        <script>
            // Simplified script assuming speak function is available globally
            const textToSpeak = {json.dumps(latest_assistant_response)};
            const voiceNameToUse = {json.dumps(st.session_state.selected_voice_name)};
            function speak(text, voiceName) {{
                const synth = window.speechSynthesis;
                const isTTSEnabled = {str(tts_enabled).lower()};
                if (!isTTSEnabled || !text) return;
                if (synth.speaking) {{ synth.cancel(); }}
                const utterThis = new SpeechSynthesisUtterance(text);
                utterThis.onerror = (event) => console.error("TTS Error:", event);
                let voices = synth.getVoices(); let voiceToUse = null;
                if (voiceName === "{DEFAULT_VOICE_NAME}") voiceToUse = voices.find(v => v.default) || voices[0];
                /* Add Mufasa heuristic find if needed */
                else voiceToUse = voices.find(v => v.name === voiceName);
                if (voiceToUse) utterThis.voice = voiceToUse;
                setTimeout(() => synth.speak(utterThis), 50);
            }}
            if (textToSpeak) {{ speak(textToSpeak, voiceNameToUse); }}
        </script>
        """, height=0 )
elif not latest_assistant_response and "last_spoken_trigger_text" in st.session_state:
     del st.session_state["last_spoken_trigger_text"]

# --- Placeholder for Gmail ---
# st.sidebar.button("Connect Gmail (Placeholder)")