# idea_critic_full.py
"""
IdeaCritic - Full version (Optimist, Critic, Evaluator, Market Analyst (RAG), Investor Bot)
Features:
 - Clarifying questions
 - Multi-round debate (streamed)
 - Final business analyst summary
 - Market Analyst using Tavily (RAG) — runs AFTER final summary
 - Investor Bot — structured numeric sub-scores + verdict + recommendations
 - Save to MongoDB and archive view
"""

import os
import re
import datetime
import requests
from dotenv import load_dotenv

import streamlit as st
from pymongo import MongoClient, DESCENDING
from pymongo.server_api import ServerApi

# Optional: Gemini (Google generative AI) — keep to match your original setup
import google.generativeai as genai

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="IdeaCritic", page_icon="🚀", layout="wide")

# ---------------------------
# Helpers: initialize Gemini
# ---------------------------
@st.cache_resource
def init_gemini():
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY missing in .env. Add your Gemini API key.")
        st.stop()
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # model selection — keep same name you used earlier; change if needed
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        st.stop()

# ---------------------------
# Helpers: initialize Mongo
# ---------------------------
@st.cache_resource
def init_mongo():
    if not MONGO_CONNECTION_STRING:
        st.error("MONGO_CONNECTION_STRING missing in .env. Add your Mongo URI.")
        st.stop()
    try:
        client = MongoClient(MONGO_CONNECTION_STRING, server_api=ServerApi("1"))
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        st.stop()

# instantiate resources
gemini_model = init_gemini()
mongo_client = init_mongo()
db = mongo_client["ideacritic_db"]
debates_collection = db["debates"]
market_cache = db["market_cache"]

# ---------------------------
# RAG: fetch market data using Tavily
# ---------------------------
def fetch_market_trends(query: str, max_results: int = 5) -> str:
    """
    Query Tavily API and return concatenated snippets.
    Caches results in Mongo to avoid repeated queries.
    (No fallback stubbing — if API fails, we return empty or an error string.)
    """
    if not TAVILY_API_KEY:
        return "⚠️ TAVILY_API_KEY missing in .env — cannot fetch market data."

    # simple cache by exact query
    cached = market_cache.find_one({"query": query})
    if cached:
        # Return cached results (assume freshness OK — can add TTL logic later)
        return cached.get("results", "")

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "num_results": max_results},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        results = [item.get("content", "") for item in data.get("results", []) if item.get("content")]
        combined = "\n\n".join(results[:max_results])

        # cache into mongo
        market_cache.insert_one({
            "query": query,
            "results": combined,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc)
        })

        return combined
    except Exception as e:
        # bubble up a useful error string (no stubbing)
        return f"Error fetching market data: {e}"

# ---------------------------
# Streaming wrapper for Gemini generate_content
# ---------------------------
def stream_response_generator(prompt: str):
    """
    Yield streaming chunks from Gemini's generate_content.
    Each yielded string should be displayed incrementally.
    """
    try:
        stream = gemini_model.generate_content(prompt, stream=True)
        buffer = ""
        for chunk in stream:
            # chunk may have .text attribute containing partial text
            if getattr(chunk, "text", None):
                buffer += chunk.text
                yield chunk.text
        # (optionally) yield a marker or final buffer at end
    except Exception as e:
        yield f"[Model Error] {e}"

# ---------------------------
# Clarifying questions
# ---------------------------
@st.cache_data
def generate_clarifying_questions(idea_title: str, idea_desc: str):
    prompt = f"""
You are a practical startup mentor. A founder provided this idea:
Title: {idea_title}
Description: {idea_desc}

Generate exactly 3–5 clarifying questions to better understand this idea.
- Output exactly as a numbered list:
  1. <question>
  2. <question>
  ...
- Focus on market, target segment, feasibility, differentiation, and execution.
- Do not add extra text.
"""
    try:
        r = gemini_model.generate_content(prompt)
        raw = r.text.strip()
        # parse numbered lines
        lines = [line.strip() for line in raw.splitlines() if re.match(r"^\d+\.", line.strip())]
        return lines if lines else [raw]
    except Exception as e:
        return [f"Error generating questions: {e}"]

# ---------------------------
# Persona response handler
# ---------------------------
def get_agent_response(persona: str, idea: str, last_statement: str = None):
    """
    Returns a generator for persona responses.
    Personas: Optimist, Critic, Evaluator/Business Analyst, Market Analyst, Investor
    Market Analyst uses RAG data from Tavily.
    Investor returns structured numeric outputs.
    """
    # Market Analyst (RAG)
    if persona == "Market Analyst":
        search_query = f"Recent market trends, competitors, pricing, funding signals for: {idea}"
        market_data = fetch_market_trends(search_query)
        prompt = f"""
You are a Market Analyst. Use the following retrieved market snippets (RAG) to produce an evidence-backed summary.

Startup Idea:
{idea}

Recent Market Data:
{market_data}

Task:
- Provide a short evidence-based summary (3-5 lines).
- Highlight competitor signals, funding/traction notes, market growth or saturation, and GTM/pricing cues.
- Keep output factual and concise.
"""
        return stream_response_generator(prompt)

    # Investor persona
    if persona == "Investor":
        # structured numeric evaluation
        prompt = f"""
You are an experienced early-stage investor. Evaluate the following startup idea and provide:

1) Five sub-scores on a 0-10 scale (integers or one decimal) with a one-line justification each:
   - Market Potential
   - Innovation
   - Scalability
   - Team Feasibility
   - Risk (10 = very low risk, 0 = very high risk)

2) Compute a weighted overall score (0-100) using weights:
   Market Potential 30%, Innovation 25%, Scalability 20%, Team Feasibility 15%, Risk 10%

3) Provide a short verdict (choose one: "Strong Buy", "Consider with Caution", "Not Investable Yet")

4) Give 3 concise next-step recommendations for the founder.

Startup Idea:
{idea}

Format strictly as:
Market Potential: <score> — <justification>
Innovation: <score> — <justification>
Scalability: <score> — <justification>
Team Feasibility: <score> — <justification>
Risk: <score> — <justification>
Weighted Score (0-100): <score>
Verdict: <verdict>
Recommendations:
1. <rec1>
2. <rec2>
3. <rec3>
"""
        return stream_response_generator(prompt)

    # Default personas: Optimist, Critic, Business Analyst/Evaluator
    if not last_statement:
        prompt = f"You are a startup {persona}. Analyze the idea: '{idea}' in 2–3 concise bullet points. Be specific and actionable."
    else:
        prompt = f"You are a startup {persona}. The idea: '{idea}'. The last statement was: '{last_statement}'. Respond directly in 2–3 clear points."

    return stream_response_generator(prompt)

# ---------------------------
# Final business analyst summary generator
# ---------------------------
def get_summary(idea: str, full_transcript: str):
    prompt = f"""
You are an expert Business Analyst. Given the following discussion transcript for '{idea}', write:

- A short actionable paragraph (3-4 sentences)
- Then 3 key actionable bullet points

Transcript:
{full_transcript}
"""
    return stream_response_generator(prompt)

# ---------------------------
# UI: New Analysis Page
# ---------------------------
# ---------------------------
# UI: New Analysis Page (fixed)
# ---------------------------
def show_new_analysis_page():
    st.title("🚀 New Idea Analysis")

    # initialize session state if needed
    if "clarifying_questions" not in st.session_state:
        st.header("Step 1: Describe your startup idea")
        st.session_state["idea_title"] = st.text_input(
            "Enter a short title for your idea", 
            placeholder="e.g., EcoSnap — AI litter detection"
        )
        st.session_state["idea_desc"] = st.text_area(
            "Describe your startup idea in detail", 
            placeholder="My startup will ...", 
            height=180
        )

        if st.button("Proceed", type="primary"):
            if st.session_state["idea_title"] and st.session_state["idea_desc"]:
                with st.spinner("Generating clarifying questions..."):
                    st.session_state["clarifying_questions"] = generate_clarifying_questions(
                        st.session_state["idea_title"], 
                        st.session_state["idea_desc"]
                    )
                st.session_state["answers"] = {}
                st.rerun()
            else:
                st.error("Please fill both title and description.")

    else:
        st.header("Step 2: Answer the clarifying questions")
        for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
            q_clean = re.sub(r'^\d+\.\s*', '', q)
            st.session_state["answers"][f"Q{i}"] = st.text_area(f"**{q_clean}**", key=f"q{i}", height=80)

        st.divider()
        st.header("Step 3: Start the analysis")
        num_rounds = st.slider("How many rounds should the discussion be?", 1, 5, 3)

        if st.button("Start Analysis", type="primary"):
            idea_full_context = st.session_state["idea_desc"] + "\n\n---Clarifying Details---\n"
            for i, q in enumerate(st.session_state["clarifying_questions"], start=1):
                q_clean = re.sub(r'^\d+\.\s*', '', q)
                idea_full_context += f"Q: {q_clean}\nA: {st.session_state['answers'].get(f'Q{i}', 'Not answered.')}\n"

            # transcript collectors
            conversation_history_for_db = ""
            full_transcript_text = ""
            last_response = ""

            st.subheader("💬 Live Discussion Transcript")

            # For each round, Optimist -> Critic
            for r in range(num_rounds):
                round_no = r + 1
                with st.container():
                    st.markdown(f"#### Round {round_no}")

                    # Optimist
                    st.markdown("**Optimist's Turn:**")
                    optimist_placeholder = st.empty()
                    with st.spinner("Optimist is thinking..."):
                        optimist_gen = get_agent_response("Optimist", idea_full_context, last_response)
                        optimist_response = ""
                        for chunk in optimist_gen:
                            optimist_response += chunk
                            optimist_placeholder.markdown(optimist_response)

                    conversation_history_for_db += f"\nRound {round_no} - Optimist: {optimist_response}"
                    full_transcript_text += f"\nRound {round_no} - Optimist: {optimist_response}"
                    last_response = optimist_response

                    st.divider()

                    # Critic
                    st.markdown("**Critic's Turn:**")
                    critic_placeholder = st.empty()
                    with st.spinner("Critic is thinking..."):
                        critic_gen = get_agent_response("Critic", idea_full_context, last_response)
                        critic_response = ""
                        for chunk in critic_gen:
                            critic_response += chunk
                            critic_placeholder.markdown(critic_response)

                    conversation_history_for_db += f"\nRound {round_no} - Critic: {critic_response}"
                    full_transcript_text += f"\nRound {round_no} - Critic: {critic_response}"
                    last_response = critic_response

            # After rounds — final business analyst summary
            st.divider()
            st.subheader("--- Final Business Analyst Summary ---")
            summary_placeholder = st.empty()
            with st.spinner("Drafting the final summary..."):
                summary_gen = get_summary(idea_full_context, full_transcript_text)
                final_summary = ""
                for chunk in summary_gen:
                    final_summary += chunk
                    summary_placeholder.markdown(final_summary)

            conversation_history_for_db += f"\nFinal Summary: {final_summary}"
            full_transcript_text += f"\nFinal Summary: {final_summary}"
            last_response = final_summary

            # MARKET ANALYST (runs AFTER final summary)
            st.divider()
            st.subheader("🌍 Market Analyst (RAG-powered) — Evidence-backed insight")
            market_placeholder = st.empty()
            with st.spinner("Fetching market data and producing market analyst insight..."):
                market_gen = get_agent_response("Market Analyst", idea_full_context, last_response)
                market_insight = ""
                for chunk in market_gen:
                    market_insight += chunk
                    market_placeholder.markdown(market_insight)

            conversation_history_for_db += f"\nMarket Analyst: {market_insight}"
            full_transcript_text += f"\nMarket Analyst: {market_insight}"
            last_response = market_insight

            # INVESTOR BOT
            st.divider()
            st.subheader("💼 Investor Bot — Investment Score & Recommendations")
            investor_placeholder = st.empty()
            with st.spinner("Investor Bot is evaluating the idea..."):
                investor_gen = get_agent_response(
                    "Investor", 
                    idea_full_context + "\n\nMarket Analyst insight:\n" + market_insight, 
                    last_response
                )
                investor_output = ""
                for chunk in investor_gen:
                    investor_output += chunk
                    investor_placeholder.markdown(investor_output)

            conversation_history_for_db += f"\nInvestor Bot: {investor_output}"
            full_transcript_text += f"\nInvestor Bot: {investor_output}"

            # Save to MongoDB
            try:
                doc = {
                    "idea_title": st.session_state.get("idea_title", "Untitled"),
                    "idea_description": st.session_state.get("idea_desc", ""),
                    "clarifying_answers": st.session_state.get("answers", {}),
                    "debate_transcript": conversation_history_for_db.strip(),
                    "final_summary": final_summary,
                    "market_insight": market_insight,
                    "investor_output": investor_output,
                    "created_at": datetime.datetime.now(datetime.timezone.utc)
                }
                res = debates_collection.insert_one(doc)
                st.success(f"💾 Analysis saved! Document ID: {res.inserted_id}")
            except Exception as e:
                st.error(f"Failed to save to DB: {e}")

# ---------------------------
# UI: History Page
# ---------------------------
def show_analysis_history_page(all_analyses):
    st.title("📚 Analysis Archive")
    if not all_analyses:
        st.warning("Your archive is empty.")
        return

    st.divider()
    total = len(all_analyses)
    latest = all_analyses[0]["created_at"] if total > 0 else None
    c1, c2 = st.columns(2)
    c1.metric("Total Analyses", total)
    if latest:
        c2.metric("Most Recent", latest.strftime("%B %d, %Y at %I:%M %p"))
    st.divider()

    for analysis in all_analyses:
        with st.expander(f"{analysis.get('idea_title','Untitled')} — {analysis.get('created_at').strftime('%b %d, %Y')}"):
            st.markdown(f"**Title:** {analysis.get('idea_title')}")
            st.markdown(f"**Created at:** {analysis.get('created_at').strftime('%B %d, %Y at %I:%M %p')}")
            st.markdown("**Final Summary:**")
            st.write(analysis.get("final_summary", ""))
            st.markdown("**Market Insight:**")
            st.write(analysis.get("market_insight", ""))
            st.markdown("**Investor Output:**")
            st.write(analysis.get("investor_output", ""))
            with st.expander("Full Debate Transcript"):
                st.text(analysis.get("debate_transcript", ""))

# ---------------------------
# Sidebar & Routing
# ---------------------------
st.sidebar.markdown("## 🚀 IdeaCritic")
st.sidebar.divider()
pages = ["New Analysis", "Analysis History"]
def on_page_change():
    if st.session_state.get("radio_nav") == "New Analysis":
        for key in ["clarifying_questions", "idea_title", "idea_desc", "answers", "selected_debate_id"]:
            if key in st.session_state:
                del st.session_state[key]

selected = st.sidebar.radio("Main Menu", pages, key="radio_nav", on_change=on_page_change, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.subheader("App Status")
try:
    total_docs = debates_collection.count_documents({})
    st.sidebar.metric("Saved Analyses", total_docs)
    st.sidebar.success("✅ MongoDB connected")
except Exception:
    st.sidebar.error("DB connection error")

# Routing
if selected == "New Analysis":
    show_new_analysis_page()
else:
    all_docs = list(debates_collection.find().sort("created_at", DESCENDING))
    show_analysis_history_page(all_docs)
