import streamlit as st
import os
import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi
import re

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Page Configuration ---
st.set_page_config(
    page_title="IdeaCritic (LangChain)",
    page_icon="🚀",
    layout="wide"
)

# --- Configurations and Initializations ---
load_dotenv()

@st.cache_resource
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in .env file. Please add it.")
        st.stop()
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, streaming=True)
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

llm = get_llm()
mongo_client = get_mongo_connection()
db = mongo_client['ideacritic_db']
debates_collection = db['debates']

# --- Prompt Templates ---
clarify_prompt = PromptTemplate(
    input_variables=["title", "desc"],
    template="""
    You are a startup mentor. A founder provided this idea:
    Title: {title}
    Description: {desc}

    Generate exactly 3–5 clarifying questions to better understand this idea.
    - Output format must be strictly a numbered list like:
      1. <question>
      2. <question>
      ...
    - Keep questions short, clear, and specific.
    - Focus on market, target audience, feasibility, uniqueness, and execution.
    """
)

optimist_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
    You are a startup Optimist. The startup idea is: "{idea}".

    Debate so far:
    {transcript}

    Now respond with exactly 3 concise bullet points.
    - Defend against the Critic’s previous objections where possible.
    - Highlight new strengths and opportunities.
    - Keep the points short, sharp, and positive.
    """
)

critic_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
    You are a startup Critic. The startup idea is: "{idea}".

    Debate so far:
    {transcript}

    The Optimist has just spoken. Now respond point-by-point:
    - Mirror each Optimist bullet point with a Critic counterpoint.
    - Keep the order aligned (1 vs 1, 2 vs 2, etc).
    - Use short and sharp sentences that directly challenge optimism.
    """
)

summary_prompt = PromptTemplate(
    input_variables=["idea", "transcript"],
    template="""
    You are an expert Business Analyst. You have a discussion transcript for the startup idea "{idea}".

    Discussion Transcript:
    ---
    {transcript}
    ---

    Write a final actionable summary:
    - First, a short paragraph with your verdict.
    - Then 3 key bullet points.
    """
)

# --- Chains ---
clarify_chain = LLMChain(llm=llm, prompt=clarify_prompt)
optimist_chain = LLMChain(llm=llm, prompt=optimist_prompt)
critic_chain = LLMChain(llm=llm, prompt=critic_prompt)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# --- Core Functions ---
@st.cache_data
def generate_clarifying_questions(title, desc):
    raw_text = clarify_chain.run({"title": title, "desc": desc})
    questions = [q.strip() for q in raw_text.split("\n") if re.match(r'^\d+\.', q.strip())]
    return questions if questions else [raw_text]

def get_agent_response(chain, idea, transcript):
    return chain.run({"idea": idea, "transcript": transcript})

def get_summary(idea, transcript):
    return summary_chain.run({"idea": idea, "transcript": transcript})

# --- Page 1: New Analysis Page ---
def show_new_analysis_page():
    st.title("🚀 New Idea Analysis (LangChain)")

    if "clarifying_questions" not in st.session_state:
        st.header("Step 1: Describe your startup idea")
        startup_idea_title = st.text_input("Enter a short title", placeholder="e.g., AI-powered fitness coach")
        startup_idea_desc = st.text_area("Describe your startup idea", height=150)

        if st.button("Proceed", type="primary"):
            if startup_idea_title and startup_idea_desc:
                with st.spinner("Generating clarifying questions..."):
                    st.session_state["clarifying_questions"] = generate_clarifying_questions(startup_idea_title, startup_idea_desc)
                st.session_state["idea_title"] = startup_idea_title
                st.session_state["idea_desc"] = startup_idea_desc
                st.session_state["answers"] = {}
                st.rerun()
            else:
                st.error("Please enter both title and description.")
    else:
        st.header("Step 2: Answer the clarifying questions")
        for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
            q_cleaned = re.sub(r'^\d+\.\s*', '', q)
            st.session_state["answers"][f"Q{i}"] = st.text_area(f"**{q_cleaned}**", key=f"q{i}")

        st.divider()
        st.header("Step 3: Start the analysis")
        num_rounds = st.slider("How many rounds should the discussion be?", 1, 5, 3)

        if st.button("Start Analysis", type="primary"):
            idea_full_context = st.session_state["idea_desc"] + "\n\n---Clarifying Details---\n"
            for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
                q_cleaned = re.sub(r'^\d+\.\s*', '', q)
                idea_full_context += f"Q: {q_cleaned}\nA: {st.session_state['answers'].get(f'Q{i}', 'Not answered.')}\n"

            transcript = ""

            st.subheader("💬 Live Discussion Transcript")
            for i in range(num_rounds):
                round_number = i + 1
                with st.container():
                    st.markdown(f"#### Round {round_number}")
                    
                    st.markdown("Optimist's Turn:")
                    with st.spinner("Optimist is thinking..."):
                        optimist_response = get_agent_response(optimist_chain, idea_full_context, transcript)
                        st.markdown(optimist_response)
                    transcript += f"\nRound {round_number} - Optimist: {optimist_response}"

                    st.divider()
                    st.markdown("Critic's Turn:")
                    with st.spinner("Critic is thinking..."):
                        critic_response = get_agent_response(critic_chain, idea_full_context, transcript)
                        st.markdown(critic_response)
                    transcript += f"\nRound {round_number} - Critic: {critic_response}"

            st.divider()
            st.subheader("--- Final Business Analyst Summary ---")
            with st.spinner("Drafting final summary..."):
                final_summary = get_summary(idea_full_context, transcript)
                st.markdown(final_summary)

            try:
                doc = {
                    "idea_title": st.session_state["idea_title"],
                    "idea_description": st.session_state["idea_desc"],
                    "clarifying_answers": st.session_state["answers"],
                    "debate_transcript": transcript.strip(),
                    "final_summary": final_summary,
                    "created_at": datetime.datetime.now(datetime.timezone.utc)
                }
                result = debates_collection.insert_one(doc)
                st.success(f"💾 Analysis saved! Document ID: {result.inserted_id}")
            except Exception as e:
                st.error(f"❌ Failed to save analysis: {e}")

# --- Page 2: History Page ---
def show_analysis_history_page(all_analyses):
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
            st.subheader("Final Summary")
            st.markdown(selected_analysis['final_summary'])
            st.subheader("Full Breakdown")
            with st.expander("Original Idea"):
                st.write(selected_analysis['idea_description'])
            with st.expander("Clarifying Answers"):
                st.write(selected_analysis.get('clarifying_answers', {}))
            with st.expander("Transcript"):
                st.text(selected_analysis['debate_transcript'])
    else:
        st.title("📚 Analysis Archive")
        if not all_analyses:
            st.warning("Archive is empty.")
            return
        st.subheader("All Saved Analyses")
        for analysis in all_analyses:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"##### {analysis['idea_title']}")
                    st.caption(f"Created on: {analysis['created_at'].strftime('%B %d, %Y %I:%M %p')}")
                with col2:
                    if st.button("View Full Report", key=f"view_{analysis['_id']}"):
                        st.session_state.selected_debate_id = str(analysis['_id'])
                        st.rerun()
                st.write(analysis.get('final_summary', '')[:200] + "...")

# --- Sidebar Navigation ---
st.sidebar.markdown("## 🚀 IdeaCritic (LangChain)")
page_options = ["New Analysis", "Analysis History"]

def on_page_change():
    if st.session_state.radio_nav == "New Analysis":
        for key in ["clarifying_questions", "idea_title", "idea_desc", "answers", "selected_debate_id"]:
            if key in st.session_state:
                del st.session_state[key]

selected_page = st.sidebar.radio("Main Menu", page_options, key="radio_nav", on_change=on_page_change)
st.sidebar.divider()

try:
    total_analyses = debates_collection.count_documents({})
    st.sidebar.metric("Total Analyses", total_analyses)
    st.sidebar.success("✅ MongoDB Connected")
except Exception:
    st.sidebar.error("DB connection error.")

# --- Page Routing ---
if selected_page == "New Analysis":
    show_new_analysis_page()
elif selected_page == "Analysis History":
    all_analyses = list(debates_collection.find().sort("created_at", DESCENDING))
    show_analysis_history_page(all_analyses)