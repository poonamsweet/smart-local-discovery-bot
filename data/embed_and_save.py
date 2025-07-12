import json
import pickle
from sentence_transformers import SentenceTransformer

def embed_and_save(json_path, pkl_path, text_func):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [text_func(item) for item in data]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    with open(pkl_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "data": data}, f)
    print(f"Embeddings and data saved to {pkl_path}")

# Restaurants
embed_and_save(
    "data/restaurants.json",
    "data/restaurants_embeddings.pkl",
    lambda item: f"{item['name']} {item['address']} {item.get('cuisine', '')}"
)

# Dentists
embed_and_save(
    "data/dentists.json",
    "data/dentists_embeddings.pkl",
    lambda item: f"{item['name']} {item['address']}"
)

# Events
embed_and_save(
    "data/events.json",
    "data/events_embeddings.pkl",
    lambda item: f"{item['name']} {item['address']} {item.get('date', '')} {item.get('category', '')}"
)