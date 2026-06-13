# ✈️ Global Wanderer AI

An AI-powered Travel Assistant built using **Streamlit, FastAPI, LangChain, Gemini, SQLite, and FAISS RAG**.

## 🚀 Features

- 🤖 AI Travel Agent (Gemini 2.5 Flash)
- 🌤 Real-time Weather Information
- 🏨 Hotel Recommendations
- 🍽 Restaurant Discovery
- ✈ Flight Search
- 🗺 Tourist Attractions & Maps
- 🧠 Conversation Memory
- 📚 RAG with FAISS Vector Database
- 📊 Analytics Dashboard
- 🔍 Search & Filter Functionality
- 💾 SQLite Database Integration
- ⚡ FastAPI Backend Support
- 📄 Knowledge Base PDF Upload & Retrieval

## 🛠 Tech Stack

- Streamlit
- FastAPI
- LangChain
- Gemini 2.5 Flash
- SQLite
- FAISS
- HuggingFace Embeddings
- OpenWeatherMap API
- SerpAPI
- OpenStreetMap

## 📂 Project Structure

```text
project/
│
├── agent.py
├── agent_ui1.py
├── api.py
├── db.py
├── build_vector_db.py
│
├── pages/
│   ├── Dashboard.py
│   └── Knowledge_Base.py
│
├── knowledge_base/
├── faiss_index/
└── requirements.txt

```
#⚙️ Installation
-git clone <repo-url>
-cd project

-pip install -r requirements.txt

-Create a .env file:

-GOOGLE_API_KEY=your_key
-SERP_API_KEY=your_key
-WEATHER_API_KEY=your_key
#▶ Run Application
-Streamlit UI
-streamlit run agent_ui1.py
-FastAPI Server
-uvicorn api:app --reload
#📚 Build Knowledge Base

-Add PDFs to:

-knowledge_base/

-Then run:

-python build_vector_db.py
#🎯 Example Queries
-Plan a 3-day trip to Goa
-Flights from Ahmedabad to Delhi
-Best hotels in Jaipur
-Weather in Manali
-Show attractions in Mumbai on map
#📊 Dashboard

-Monitor:

-Total Conversations
-Tool Usage Statistics
-Average Response Time
-Search & Filter Analytics
#🌐 Deployment

-Deploy on:

-Streamlit Community Cloud
## Live Demo

[Launch Application](https://ai-rag-travel-agent-lzzrdms8txtemmi6lppv7c.streamlit.app/)
