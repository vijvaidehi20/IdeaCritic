# 🚀 IdeaCritic

An AI-Powered Multi-Agent Startup Idea Evaluation Platform built with **Streamlit**, **Google Gemini**, and **MongoDB**.

---

## 📌 Overview

**IdeaCritic** bridges the gap between raw startup ideas and structured evaluation. It leverages a multi-agent AI system—simulating an Optimist, Critic, Business Analyst, Market Analyst, and Early-Stage Investor—to provide founders with a realistic, data-driven, and actionable perspective on their new ventures.

By identifying blind spots, providing market intelligence, and delivering an investor readiness score, IdeaCritic transforms vague concepts into well-assessed business propositions.

## ✨ Key Features

1. **Clarifying Questions Engine**: Before evaluation, the AI asks 3-5 tailored clarifying questions to deeply understand your specific market and execution strategy.
2. **Multi-Agent AI Debate**: Watch a live Streamlit debate where an **Optimist** highlights strengths and a **Critic** probes for flaws across multiple customizable discussion rounds.
3. **Business Analyst Summary**: Synthesizes the entire debate transcript into concise, actionable takeaways and next steps.
4. **Market Analyst (RAG)**: Integrates with the **Tavily API** to automatically search the web for the latest market trends, competitor insights, and funding signals, grounding the evaluation in actual market conditions.
5. **Investor Bot**: Computes a weighted overall score out of 100 based on Market Potential, Innovation, Scalability, Team Feasibility, and Risk. It delivers a firm verdict ("Strong Buy", "Consider with Caution", etc.) and practical recommendations.
6. **Persistent Archive**: All debates, final summaries, and scores are automatically saved to a **MongoDB** database so you can revisit past ideas using the 'Analysis History' page.

## ⚙️ Tech Stack

- **Frontend / Fullstack Framework**: [Streamlit](https://streamlit.io/)
- **Large Language Model (LLM)**: [Google Gemini 2.5 Flash](https://ai.google.dev/) via `google-generativeai`
- **Real-Time Market Data / RAG**: [Tavily API](https://tavily.com/)
- **Database**: [MongoDB](https://www.mongodb.com/) (using `pymongo`)

## ▶️ Setup and Installation

### 1. Prerequisites

- Python 3.9+
- A [Google Gemini API Key](https://aistudio.google.com/app/apikey)
- A [Tavily API Key](https://app.tavily.com/)
- A [MongoDB Cluster URI](https://www.mongodb.com/products/platform/atlas-database)

### 2. Clone the Repository

```bash
git clone https://github.com/vijvaidehi20/IdeaCritic.git
cd IdeaCritic
```

### 3. Install Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the root directory of the project and add your API keys and connection string:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
MONGO_CONNECTION_STRING=your_mongodb_connection_uri_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Open your browser to the local address provided (usually `http://localhost:8501`) to start evaluating your ideas.

---

## 🛠️ Usage Flow

1. Navigate to the **"New Analysis"** tab.
2. Enter your startup's name and a high-level description.
3. Answer the dynamically generated clarifying questions.
4. Select the number of debate rounds and initiate the analysis.
5. Watch the real-time AI debate followed by the Business Analyst, Market Analyst, and Investor modules.
6. Open the **"Analysis History"** tab from the sidebar any time to review previous evaluations.

---

*“Great ideas survive criticism. IdeaCritic ensures yours stands the test.”*
