# Smart Local Discovery Chatbot

A prototype AI chatbot that helps users discover hyperlocal services or businesses using natural language queries. This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using local vector search (sentence-transformers) and OpenAI for answer generation.

---

## Features
- **Chat-style UI** (Streamlit)
- **Semantic search** over mock data (restaurants, dentists, events) using sentence-transformers
- **RAG pipeline**: Retrieves relevant results and uses OpenAI GPT for natural language answers
- **No external vector DB required** (all embeddings stored locally)
- **Docker-ready** for easy deployment

---

## How It Works
1. **User** enters a natural language query (e.g., "Find vegetarian restaurants near MG Road, Bengaluru").
2. **Semantic Retrieval**: The query is embedded using sentence-transformers and compared (cosine similarity) to pre-computed embeddings of mock data (stored in `.pkl` files).
3. **Augmented Generation**: Top results are passed as context to OpenAI GPT, which generates a friendly, context-aware answer.
4. **Chatbot** displays the answer in the UI.

---

## Project Structure
```
├── app.py                  # Streamlit app (UI + main logic)
├── retriever.py            # Semantic search (local vector search)
├── llm.py                  # OpenAI answer generation
├── data/
│   ├── restaurants.json    # Mock data
│   ├── dentists.json       # Mock data
│   ├── events.json         # Mock data
│   ├── embed_and_save.py   # Script to generate embeddings
│   └── restaurants_embeddings.pkl # Saved embeddings
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Docker Compose config
├── .env                    # API keys (OpenAI)
└── README.md               # Project documentation
```

---

## Setup Instructions

### 1. Clone the Repo & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API Keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-key-here
```

### 3. Generate Embeddings (First Time Only)
```bash
python data/embed_and_save.py
```
This will create `data/restaurants_embeddings.pkl`.

### 4. Run the App (Locally)
```bash
streamlit run app.py
```

---

## Docker Usage

### Build the Image
```bash
docker-compose build
```

### Generate Embeddings in the Container
```bash
docker-compose run smart-local-discovery-bot python data/embed_and_save.py
```

### Start the App
```bash
docker-compose up
```

App will be available at [http://localhost:8501](http://localhost:8501)

---

## RAG Pipeline (How it works)
1. **Embeddings**: All mock data is embedded using sentence-transformers and stored locally.
2. **Semantic Search**: User query is embedded and compared to data embeddings (cosine similarity) to find top results.
3. **LLM Augmentation**: Top results are sent as context to OpenAI GPT, which generates a natural language answer.

---

## Example Queries
- Find vegetarian restaurants within 2 km of MG Road, Bengaluru
- What are the top-rated dentists near me?
- Show me trending events in Jaipur this weekend
- Is there any late-night food delivery around Indiranagar?

---

## Notes
- This project uses only local files for vector search (no Pinecone or external DB).
- You can extend the approach to other data (dentists, events) by generating embeddings for those files as well.
- For production, you can swap local vector search with Pinecone, ChromaDB, etc.

---

