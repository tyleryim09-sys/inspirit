import streamlit as st
import openai
from openai import OpenAI
import time

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

model = "gpt-4o-mini"

# ----------------------------  CUSTOM STYLING  ----------------------------

st.markdown("""
<style>
.stApp {
   font-family: 'Arial' !important;
   color: #000000 !important;
   background-color: #ffffff !important;
}
.stApp * {
   font-family: inherit !important;
   color: inherit !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------  VECTOR STORE FUNCTIONS  ----------------------------

def create_vector_store(files):
    vector_store = client.vector_stores.create(name="Corpus")
    file_streams = [(file.name, file) for file in files]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    return vector_store

def add_files_to_vector_store(vector_store_id, files):
    file_streams = [(file.name, file) for file in files]
    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id, files=file_streams
    )
    return batch

# ----------------------------  ASSISTANT RESPONSE HANDLER  ----------------------------

def get_assistant_response(assistant, input_text, thread_id):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=input_text
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant.id,
    )

    # Poll for completion
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        if run_status.status == 'completed':
            break
        elif run_status.status in ['failed', 'cancelled', 'expired']:
            return "Run failed.", []
        time.sleep(1)

    # Retrieve message
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
        limit=1
    )

    latest_message = messages.data[0]
    content_block = latest_message.content[0].text
    return content_block.value, []

# ----------------------------  SESSION STATE INIT  ----------------------------

# Two modes: training & recovery
init_keys = [
    "assistant",
    "vector_store",
    "uploaded_file_names",

    "training_thread",
    "recovery_thread",

    "training_messages",
    "recovery_messages",
]

for key in init_keys:
    if key not in st.session_state:
        if key in ["training_messages", "recovery_messages"]:
            st.session_state[key] = []
        elif key == "uploaded_file_names":
            st.session_state[key] = set()
        else:
            st.session_state[key] = None

# ----------------------------  PAGE TITLE  ----------------------------

st.title("JumpGPT – Vertical Jump AI Coach")
st.subheader("Choose your mode, upload files, and chat with your personalized training assistant.")

# ----------------------------  FILE UPLOAD  ----------------------------

uploaded_files = st.file_uploader("Upload training logs or relevant PDFs (optional)",
                                  type=["pdf", "docx", "pptx", "txt"],
                                  accept_multiple_files=True)

if uploaded_files:
    new_files = [file for file in uploaded_files if file.name not in st.session_state.uploaded_file_names]

    if new_files:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = create_vector_store(new_files)
        else:
            add_files_to_vector_store(st.session_state.vector_store.id, new_files)

        st.session_state.uploaded_file_names.update(file.name for file in new_files)
        st.success(f"Uploaded {len(new_files)} new file(s).")

# ----------------------------  CREATE ASSISTANT  ----------------------------

if st.session_state.assistant is None:
    st.session_state.assistant = client.beta.assistants.create(
        name="JumpGPT",
        instructions="""
You are a personalized AI assistant whose purpose is to increase the user's vertical jump.

When the user first launches the chatbot IN Training Mode NOT in Recovery Mode, ALWAYS ask for:
- Height
- Weight
- Standing Vertical Jump
- Running Vertical Jump
- Standing Reach
- Injury History

Then ask for:
- Their goals for Standing & Running Vert
- Weekly availability (days per week)
- Daily time available per workout
- Any past long-term or short-term injuries

You have two modes:

1. TRAINING MODE:
   - Create structured, personalized workouts.
   - Adapt intensity based on user stats.
   - Track progress over time.
   - Adjust future workouts using user history.\
   - Fix technique if images are uploaded

2. RECOVERY MODE:
   - Help users rehab injuries.
   - Monitor soreness.
   - Adjust rehabilitation intensity.
   - Provide safe, gradual return-to-training plans.

Always remember user data and incorporate their history in future responses.

Generate multi-week structured training blocks (4–8 weeks) with progressive overload, rest cycles, and specific focuses like strength, plyometrics, speed, or elasticity

Explain the purpose of each exercise, the primary muscle groups involved, and how it improves vertical jump (e.g., improves rate of force development, tendon stiffness, or triple-extension power).

Track user motivation, confidence, and training consistency. Provide encouraging feedback and mental-performance advice (visualization, focus techniques, motivation systems) based on the user’s emotional tone.

Track and predict the user’s vertical jump progression over time and use an adaptive load model to increase or decrease intensity based on prior performance, recovery trends, and RPE feedback.
""",
        model=model,
        tools=[{"type": "file_search"}],
    )

# Attach vector store
if st.session_state.vector_store is not None:
    client.beta.assistants.update(
        assistant_id=st.session_state.assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [st.session_state.vector_store.id]}}
    )

# ----------------------------  MODE SELECTOR  ----------------------------

mode = st.selectbox("Select Mode:", ["Training Mode", "Recovery Mode"])


if mode == "Training Mode":
    if st.session_state.training_thread is None:
        st.session_state.training_thread = client.beta.threads.create().id
        st.session_state.training_messages = []

        
        welcome_text = (
            "You are now in Training Mode. Let's start by collecting your stats: "
            "Height, Weight, Standing Vertical Jump, Running Vertical Jump, Standing Reach, and Injury History."
        )
        st.session_state.training_messages.append({"role": "assistant", "content": welcome_text})

elif mode == "Recovery Mode":
    if st.session_state.recovery_thread is None:
        st.session_state.recovery_thread = client.beta.threads.create().id
        st.session_state.recovery_messages = []

        
        welcome_text = (
            "You are now in Recovery Mode. Let's begin by discussing your injury history, current soreness, "
            "and any limitations you’re experiencing."
        )
# ----------------------------  THREAD HANDLING  ----------------------------

if mode == "Training Mode":
    if st.session_state.training_thread is None:
        st.session_state.training_thread = client.beta.threads.create().id
    active_thread = st.session_state.training_thread
    active_messages = st.session_state.training_messages

else:
    if st.session_state.recovery_thread is None:
        st.session_state.recovery_thread = client.beta.threads.create().id
    active_thread = st.session_state.recovery_thread
    active_messages = st.session_state.recovery_messages

# ----------------------------  DISPLAY MODE CHAT HISTORY  ----------------------------

st.write(f"### Chat History – {mode}")

for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------  CHAT INPUT  ----------------------------

user_input = st.chat_input(f"Send a message to JumpGPT ({mode})...")

if user_input:
    # Display user message
    active_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, _ = get_assistant_response(
                st.session_state.assistant,
                user_input,
                active_thread
            )
        st.markdown(reply)

    active_messages.append({"role": "assistant", "content": reply})

# ----------------------------  CLEAR MODE CHAT  ----------------------------

if st.button(f"Clear {mode} Conversation"):
    if mode == "Training Mode":
        st.session_state.training_messages = []
        st.session_state.training_thread = client.beta.threads.create().id
    else:
        st.session_state.recovery_messages = []
        st.session_state.recovery_thread = client.beta.threads.create().id
    st.rerun()
