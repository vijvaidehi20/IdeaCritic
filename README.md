# IdeaCritic 🚀  
An AI-Powered Multi-Agent Startup Idea Evaluation Platform

---

## 📌 Overview
In today’s fast-paced innovation ecosystem, countless ideas emerge daily, but only a few evolve into successful ventures. The primary reason is the lack of early-stage validation and structured feedback.

**IdeaCritic** bridges this gap through an AI-powered multi-agent evaluation platform that delivers data-driven, balanced, and actionable insights on new ideas. By simulating an expert panel using AI agents — **Optimist, Critic, Evaluator, Market Analyst (RAG), and Investor Bot** — the system provides both supportive and critical perspectives to help founders and students refine their ideas effectively.

---

## ❓ Problem Statement
Most ideas fail to progress due to the absence of structured, data-driven evaluation in their early stages. Innovators often rely on subjective opinions or limited research, which leads to:
- Overlooked risks
- Poor market fit
- Inefficient decision-making

Traditional evaluation methods are time-consuming, biased, and inconsistent.  
**IdeaCritic** solves this by delivering objective, real-time, and evidence-based feedback using AI-powered multi-agent analysis.

---

## 💡 Proposed Solution
IdeaCritic simulates a structured expert debate using autonomous AI agents:
- **Optimist** – Highlights strengths and potential
- **Critic** – Identifies flaws and risks
- **Evaluator / Business Analyst** – Assesses feasibility and execution
- **Market Analyst (RAG)** – Fetches real-time market intelligence
- **Investor Bot** – Provides investment scoring and recommendations

This holistic analysis transforms vague concepts into actionable, well-assessed ideas.

---

## ⚙️ Technical Architecture

### 🧠 AI & LLM
- **Google Gemma-3-12b-it / Gemini API** for reasoning and critique generation

### 🧩 Frameworks
- **LangChain**
- **LangChain Google GenAI** for agent orchestration

### 🌐 Frontend & Backend
- **Streamlit** for UI, live debate streaming, and interaction handling

### 📊 Market Intelligence (RAG)
- **Tavily / Serper.dev APIs** for real-time market data retrieval

### 📈 Evaluation Engine
- Custom multi-factor scoring:
  - Innovation
  - Market Fit
  - Feasibility
  - Risk
  - Scalability

### 🛠 Data Utilities
- `pandas`, `numpy`, `python-dotenv`

### 🗄 Storage & Reporting
- **MongoDB** for storing analysis history
- **ReportLab / python-docx** for PDF and Word report generation

---

## ✨ Key Features
- Multi-agent AI debate (Optimist, Critic, Evaluator, Investor)
- RAG-based real-time market analysis
- Automated investor scoring and verdicts
- Interactive Streamlit interface
- PDF / Word report generation
- Persistent storage of past analyses using MongoDB

---

## ▶️ How to Run the Project

### Prerequisites
- Python 3.9+
- MongoDB
- API Keys:
  - Google Gemini API
  - Tavily API

### Setup
```bash
git clone https://github.com/vijvaidehi20/IdeaCritic.git
cd IdeaCritic
pip install -r requirements.txt
