# AI-RAG-Travel-Agent

## Streamlit app

```bash
streamlit run agent_ui1.py
```

## FastAPI backend

Run the API locally:

```bash
uvicorn fastApi:app --reload
```

Useful endpoints:

- `GET /health` - API health check
- `POST /chat` - send a travel-agent chat message
- `GET /stats` - dashboard metrics from SQLite
- `POST /upload-pdf` - upload a PDF into `knowledge_base/`
- `POST /rebuild-db` - rebuild the FAISS vector database in the background

Example chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Plan a 2 day trip to Goa\"}"
```

Set API keys in `.env` locally or Streamlit secrets in deployment:

```env
GROQ_API_KEY=...
SERP_API_KEY=...
WEATHER_API_KEY=...
```
