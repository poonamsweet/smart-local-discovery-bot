import openai
import streamlit as st
import os
import re
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_openai_key():
    """Get OpenAI API key from Streamlit secrets or environment variable"""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("OPENAI_API_KEY")

def extract_intent_entities(query):
    """
    Use OpenAI GPT to extract intent and entities from the user query.
    Returns a dict: {category, location, date}
    """
    api_key = get_openai_key()
    if not api_key:
        print("No OpenAI API key found. Falling back to keyword extraction.")
        return fallback_extraction(query)
    
    openai.api_key = api_key
    
    system_prompt = (
        "You are an assistant that extracts structured information from user queries about local discovery. "
        "Given a user query, extract the following as JSON: "
        "category (restaurant, dentist, event, food, delivery), location (area or city), date (YYYY-MM-DD, if present, else null). "
        "If not found, set value to null. Only output JSON."
    )
    user_prompt = f"Query: {query}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        content = response["choices"][0]["message"]["content"]
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            entities = json.loads(match.group(0))
            return entities
    except Exception as e:
        print(f"OpenAI extraction failed: {e}")

    # Fallback: simple keyword extraction
    return fallback_extraction(query)

def fallback_extraction(query):
    """Fallback keyword extraction when OpenAI is not available"""
    print("Falling back to keyword extraction.")
    categories = ["restaurant", "dentist", "event", "food", "delivery"]
    locations = ["mg road", "indiranagar", "residency road", "brigade road", "bengaluru", "jaipur"]
    category = None
    location = None
    date = None
    
    for cat in categories:
        if cat in query.lower():
            category = cat
            break
    for loc in locations:
        if loc in query.lower():
            location = loc.title()
            break
    if not location and ("near me" in query.lower() or "around here" in query.lower()):
        location = "Bengaluru"
    if "this weekend" in query.lower():
        today = datetime.now()
        weekday = today.weekday()
        days_until_saturday = (5 - weekday) % 7
        event_date = today.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_saturday)
        date = event_date.strftime("%Y-%m-%d")
    else:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if match:
            date = match.group(1)
    return {"category": category, "location": location, "date": date}

def generate_answer_with_context(query, results):
    api_key = get_openai_key()
    if not api_key:
        # Return a simple response without OpenAI
        if not results:
            return "I couldn't find any relevant results for your query."
        else:
            response = "Here's what I found:\n\n"
            for i, r in enumerate(results, 1):
                response += f"{i}. {r['name']}"
                if 'address' in r:
                    response += f" - {r['address']}"
                if 'rating' in r:
                    response += f" ({r['rating']}⭐)"
                response += "\n"
            return response
    
    openai.api_key = api_key
    
    # Format results as context
    if not results:
        context = "No relevant results found."
    else:
        context = ""
        for i, r in enumerate(results, 1):
            context += f"{i}. {r['name']} - {r.get('address', '')}"
            if 'rating' in r:
                context += f" ({r['rating']}⭐)"
            if 'date' in r:
                context += f" [Date: {r['date']}]"
            context += "\n"
    
    system_prompt = (
        "You are a helpful assistant that answers user queries about local businesses and events. "
        "Use the provided context to answer the user's question in a friendly, concise way. "
        "If the context is empty or says 'No relevant results found.', say you couldn't find anything."
    )
    user_prompt = f"User query: {query}\n\nContext:\n{context}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI generation failed: {e}")
        # Return fallback response
        if not results:
            return "I couldn't find any relevant results for your query."
        else:
            response = "Here's what I found:\n\n"
            for i, r in enumerate(results, 1):
                response += f"{i}. {r['name']}"
                if 'address' in r:
                    response += f" - {r['address']}"
                if 'rating' in r:
                    response += f" ({r['rating']}⭐)"
                response += "\n"
            return response