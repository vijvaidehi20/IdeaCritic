import streamlit as st
import os
import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="IdeaCritic",
    page_icon="🚀",
    layout="wide"
)

# --- Configurations and Initializations ---
load_dotenv()

@st.cache_resource
def get_gemini_model():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
        st.stop()
    try:
        genai.configure(api_key=google_api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Failed to configure Google AI: {e}")
        st.stop()

@st.cache_resource
def get_mongo_connection():
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    if not connection_string:
        st.error("MONGO_CONNECTION_STRING not found in .env file.")
        st.stop()
    try:
        mongo_client = MongoClient(connection_string, server_api=ServerApi('1'))
        mongo_client.admin.command('ping')
        return mongo_client
    except Exception as e:
        st.error(f"❌ Connection to MongoDB failed: {e}")
        st.stop()

gemini_model = get_gemini_model()
mongo_client = get_mongo_connection()
db = mongo_client['ideacritic_db']
debates_collection = db['debates']

# --- Core AI Functions ---
def stream_response_generator(prompt):
    """Yields response tokens from the Gemini model stream."""
    try:
        response_stream = gemini_model.generate_content(prompt, stream=True)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"An error occurred with the AI model: {e}"

@st.cache_data
def generate_clarifying_questions(idea_title, idea_desc):
    prompt = f"""
    You are a startup mentor. A founder provided this idea:
    Title: {idea_title}
    Description: {idea_desc}

    Generate exactly 3–5 clarifying questions to better understand this idea.
    - Output format must be strictly a numbered list like:
      1. <question>
      2. <question>
      ...
    - Do not add any other text before or after the list.
    - Keep questions short, clear, and specific.
    - Focus on market, target audience, feasibility, uniqueness, and execution.
    """
    try:
        response = gemini_model.generate_content(prompt)
        raw_text = response.text.strip()
        # Find questions that start with a number and a dot
        questions = [q.strip() for q in raw_text.split('\n') if re.match(r'^\d+\.', q.strip())]
        return questions if questions else [raw_text] # Fallback to raw text if parsing fails
    except Exception as e:
        return [f"An error occurred: {e}"]

def get_agent_response(persona, idea, last_statement):
    if not last_statement:
        prompt = f"You are a startup {persona}. Analyze the startup idea: '{idea}'. " \
                 f"Provide your initial analysis in 2–3 concise bullet points. Be specific."
    else:
        prompt = f"You are a startup {persona}. The startup idea is: '{idea}'. The last statement was: '{last_statement}'. " \
                 f"Respond directly to the last statement with 2–3 clear bullet points. Focus on one or two strong points only."
    return stream_response_generator(prompt)

def get_summary(idea, full_transcript):
    prompt = f"You are an expert Business Analyst. You have a discussion transcript for the startup idea '{idea}'.\n\n" \
             f"Discussion Transcript:\n---\n{full_transcript}\n---\n" \
             f"Write a final actionable summary. Provide a short paragraph, then 3 key bullet points."
    return stream_response_generator(prompt)

# --- Page 1: New Analysis Page ---
def show_new_analysis_page():
    st.title("🚀 New Idea Analysis")

    # This check ensures we start fresh if the state is not set
    if "clarifying_questions" not in st.session_state:
        st.header("Step 1: Describe your startup idea")
        startup_idea_title = st.text_input("Enter a short title for your idea", placeholder="e.g., AI-powered fitness coach")
        startup_idea_desc = st.text_area("Describe your startup idea in detail", placeholder="My startup will offer personalized workout and meal plans...", height=150)

        if st.button("Proceed", type="primary"):
            if startup_idea_title and startup_idea_desc:
                with st.spinner("Generating clarifying questions..."):
                    st.session_state["clarifying_questions"] = generate_clarifying_questions(startup_idea_title, startup_idea_desc)
                st.session_state["idea_title"] = startup_idea_title
                st.session_state["idea_desc"] = startup_idea_desc
                st.session_state["answers"] = {}
                st.rerun()
            else:
                st.error("Please enter both a title and a description.")
    
    # This part runs only after questions have been generated
    else:
        st.header("Step 2: Answer the clarifying questions")
        for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
            # Remove numbering like "1. " from the start of the question for cleaner display
            q_cleaned = re.sub(r'^\d+\.\s*', '', q)
            st.session_state["answers"][f"Q{i}"] = st.text_area(f"**{q_cleaned}**", key=f"q{i}")

        st.divider()
        st.header("Step 3: Start the analysis")
        num_rounds = st.slider("How many rounds should the discussion be?", 1, 5, 3)

        if st.button("Start Analysis", type="primary"):
            # Combine original description with Q&A for a rich context
            idea_full_context = st.session_state["idea_desc"] + "\n\n---Clarifying Details---\n"
            for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
                q_cleaned = re.sub(r'^\d+\.\s*', '', q)
                idea_full_context += f"Q: {q_cleaned}\nA: {st.session_state['answers'].get(f'Q{i}', 'Not answered.')}\n"

            conversation_history_for_db = ""
            last_response = ""

            st.subheader("💬 Live Discussion Transcript")
            for i in range(num_rounds):
                round_number = i + 1
                with st.container(border=True):
                    st.markdown(f"#### Round {round_number}")
                    st.markdown("Optimist's Turn:")
                    with st.spinner("Optimist is thinking..."):
                        optimist_response = st.write_stream(get_agent_response("Optimist", idea_full_context, last_response))
                    conversation_history_for_db += f"\nRound {round_number} - Optimist: {optimist_response}"
                    last_response = optimist_response
                    st.divider()
                    st.markdown("Critic's Turn:")
                    with st.spinner("Critic is thinking..."):
                        critic_response = st.write_stream(get_agent_response("Critic", idea_full_context, last_response))
                    conversation_history_for_db += f"\nRound {round_number} - Critic: {critic_response}"
                    last_response = critic_response

            st.divider()
            st.subheader("--- Final Business Analyst Summary ---")
            with st.spinner("Drafting the final summary..."):
                final_summary = st.write_stream(get_summary(idea_full_context, conversation_history_for_db))

            try:
                doc = {"idea_title": st.session_state["idea_title"], "idea_description": st.session_state["idea_desc"], "clarifying_answers": st.session_state["answers"], "debate_transcript": conversation_history_for_db.strip(), "final_summary": final_summary, "created_at": datetime.datetime.now(datetime.timezone.utc)}
                result = debates_collection.insert_one(doc)
                st.success(f"💾 Analysis successfully saved! Document ID: {result.inserted_id}")
            except Exception as e:
                st.error(f"❌ Failed to save analysis to the database: {e}")

# --- Page 2: History Page ---
def show_analysis_history_page(all_analyses):
    # (This function is unchanged)
    if 'selected_debate_id' in st.session_state:
        selected_id = st.session_state.selected_debate_id
        selected_analysis = next((d for d in all_analyses if str(d['_id']) == selected_id), None)
        if selected_analysis:
            if st.button("⬅️ Back to Archive"):
                del st.session_state.selected_debate_id
                st.rerun()
            st.header(f"Viewing Analysis: {selected_analysis['idea_title']}")
            st.caption(f"Analyzed on: {selected_analysis['created_at'].strftime('%B %d, %Y at %I:%M %p')}")
            st.divider()
            st.subheader("Final Business Analyst Summary")
            st.markdown(selected_analysis['final_summary'])
            st.subheader("Full Analysis Breakdown")
            with st.expander("Original Idea Description"):
                st.write(selected_analysis['idea_description'])
            with st.expander("Clarifying Answers"):
                st.write(selected_analysis.get('clarifying_answers', {}))
            with st.expander("Full Discussion Transcript (Formatted)"):
                transcript = selected_analysis['debate_transcript']
                rounds = re.findall(r"Round (\d+) - (Optimist|Critic): ([\s\S]*?)(?=\nRound|\Z)", transcript)
                if rounds:
                    for round_num, persona, text in rounds:
                        with st.chat_message(name=persona.lower(), avatar="🧑‍💻" if persona == "Optimist" else "🤔"):
                            st.write(f"{persona}'s statement in Round {round_num}:"); st.markdown(text.strip())
                else: st.text(transcript)
    else:
        st.title("📚 Analysis Archive")
        if not all_analyses: st.warning("Your archive is empty."); return
        st.divider()
        total_analyses = len(all_analyses); latest_analysis_date = all_analyses[0]['created_at']
        col1, col2 = st.columns(2); col1.metric("Total Analyses", total_analyses); col2.metric("Most Recent", latest_analysis_date.strftime("%B %d, %Y"))
        st.divider()
        st.subheader("All Saved Analyses")
        for analysis in all_analyses:
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"##### {analysis['idea_title']}"); st.caption(f"Created on: {analysis['created_at'].strftime('%B %d, %Y at %I:%M %p')}")
                with col2:
                    if st.button("View Full Report", key=f"view_{analysis['_id']}"):
                        st.session_state.selected_debate_id = str(analysis['_id']); st.rerun()
                summary_preview = analysis.get('final_summary', 'No summary available.')
                st.write(summary_preview[:200] + "...")

# --- Sidebar & Navigation (WITH THE FIX) ---
st.sidebar.markdown("## 🚀 IdeaCritic")
st.sidebar.divider()
page_options = ["New Analysis", "Analysis History"]

# This callback function will clear the state when the page changes to "New Analysis"
def on_page_change():
    if st.session_state.radio_nav == "New Analysis":
        keys_to_reset = ["clarifying_questions", "idea_title", "idea_desc", "answers", "selected_debate_id"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

selected_page = st.sidebar.radio("Main Menu", page_options, key="radio_nav", 
                                 on_change=on_page_change, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.subheader("App Stats")
try:
    total_analyses = debates_collection.count_documents({})
    st.sidebar.metric("Total Analyses Saved", total_analyses)
    st.sidebar.success("✅ Connected to MongoDB!")
except Exception: st.sidebar.error("DB connection error.")

# --- Page Routing ---
if selected_page == "New Analysis":
    show_new_analysis_page()
elif selected_page == "Analysis History":
    all_analyses = list(debates_collection.find().sort("created_at", DESCENDING))
    show_analysis_history_page(all_analyses)