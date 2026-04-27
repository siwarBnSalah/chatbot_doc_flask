# 🚀 Chatbot RAG Intelligent

<p align="center">
  <img src="https://img.shields.io/badge/Project-PFE-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RAG-Enabled-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Agents-AI-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Version-version--groq-black?style=flat-square"/>
</p>

---

## 🌟 Overview

> A next-generation **AI Chatbot system** powered by **RAG (Retrieval-Augmented Generation)**, enhanced with **Agent-based reasoning** and accelerated using **Groq LLM**.

This project transforms traditional document Q&A into an **intelligent multi-agent system** capable of reasoning, filtering, and optimizing responses.

---

## 🧠 Key Idea

👉 Instead of a simple chatbot:


User → LLM → Answer ❌


We built:


User → Agents → Retrieval → Groq LLM → Validation → Smart Answer ✅


---

## 🏗️ System Architecture

```text
User
  ↓
RoutingAgent (filter irrelevant queries)
  ↓
QueryReformulationAgent (optimize question)
  ↓
Vector Search (ChromaDB)
  ↓
Groq LLM (fast generation)
  ↓
CodeValidationAgent (verify output)
  ↓
Final Answer
⚙️ Installation
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
git checkout version-groq
🐍 Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
📦 Install Dependencies
pip install -r requirements.txt
▶️ Run Project
python app.py

or

streamlit run app.py
📂 Project Structure
chatbot_doc_flask/
│
├── app.py                  # Main interface
├── chatbot.py              # RAG pipeline
├── agents.py              # AI Agents logic
├── test_groq.py           # Groq testing
├── evaluation_reelle.py   # Evaluation system
├── requirements.txt
│
├── vector_db/            # ChromaDB storage
├── evaluation_output/    # Evaluation results
├── data_propre/          # Documents dataset
🤖 Features
🔹 Core RAG System
Semantic search (Embeddings)
Vector database (ChromaDB)
Context-aware generation
🔹 AI Agents Layer
🧭 RoutingAgent → filters irrelevant questions
✍️ QueryReformulationAgent → improves queries
🧪 CodeValidationAgent → validates outputs
🔹 LLM Engine
⚡ Groq API (Ultra-fast inference)
Context-aware response generation
🧪 Pipeline Flow
Question
   ↓
Routing Agent
   ↓
Query Optimization
   ↓
Vector Retrieval
   ↓
Groq LLM Generation
   ↓
Validation
   ↓
Final Answer
🛠️ Tech Stack
<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Groq-FF6B00?style=for-the-badge"/> <img src="https://img.shields.io/badge/ChromaDB-00C896?style=for-the-badge"/> <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit"/> <img src="https://img.shields.io/badge/NLP-DeepLearning-blueviolet?style=for-the-badge"/> </p>
📊 Evaluation System
Accuracy measurement
Response quality analysis
Visual dashboards
Real evaluation reports


