import streamlit as st
from retriever import get_local_results
from llm import extract_intent_entities
from retriever import semantic_search_local
from llm import generate_answer_with_context

st.set_page_config(page_title="Smart Local Discovery Chatbot", page_icon="🤖")

st.title("🤖 Smart Local Discovery Chatbot")
st.write("Ask me to find local businesses, services, or events near you!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! How can I help you discover something nearby today?"}
    ]

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
if prompt := st.chat_input("Type your query here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 1. Intent extraction
    intent = extract_intent_entities(prompt)
    print("Intent:", intent)  # Debug ke liye

    # 2. Semantic retrieval (local RAG)
    if intent.get("category") == "dentist":
        results = semantic_search_local(prompt, "data/dentists_embeddings.pkl", top_k=3)
    elif intent.get("category") == "event":
        results = semantic_search_local(prompt, "data/events_embeddings.pkl", top_k=3)
    else:
        results = semantic_search_local(prompt, "data/restaurants_embeddings.pkl", top_k=3)

    # 3. Augmented generation (LLM)
    response = generate_answer_with_context(prompt, results)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)