import json
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np


def get_local_results(intent_dict):
    category = intent_dict.get("category")
    location = intent_dict.get("location")
    date = intent_dict.get("date")
    data_file = None
    results = []

    if category == "restaurant":
        data_file = os.path.join("data", "restaurants.json")
    elif category == "dentist":
        data_file = os.path.join("data", "dentists.json")
    elif category == "event":
        data_file = os.path.join("data", "events.json")
    else:
        return []

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return []

    # Filter by location
    if location:
        data = [item for item in data if location.lower() in item.get("address", "").lower() or location.lower() in item.get("location", "").lower()]
    # Filter by date (for events)
    if category == "event" and date:
        data = [item for item in data if item.get("date") == date]

    # Sort by rating (if available)
    if data and "rating" in data[0]:
        data = sorted(data, key=lambda x: x["rating"], reverse=True)

    return data[:3] 

def semantic_search_local(query, embedding_file="data/restaurants_embeddings.pkl", top_k=3):
    # Load embeddings and data
    with open(embedding_file, "rb") as f:
        obj = pickle.load(f)
    embeddings = obj["embeddings"]
    data = obj["data"]

    # Embed the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode([query])[0]

    # Compute cosine similarity
    similarities = np.dot(embeddings, query_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [data[i] for i in top_indices]
    return results