from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from backend.agent import agent_executor
from backend.pipeline import rag_chain, vector_db
import os
import random

load_dotenv()

app = Flask(
    __name__,
    template_folder="ui/templates",
    static_folder="ui/static"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not vector_db:
        return jsonify({"response": "Vector database not initialized."})

    user_input = request.json.get("query", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid question."})

    user_input_lower = user_input.lower()

    # Custom name response
    if "your name" in user_input_lower or "what is your name" in user_input_lower or "who are you" in user_input_lower:
        return jsonify({"response": "I’m ScandDoc AI — your local document reading assistant. I don’t have a name like humans do, but I’m here to help you understand your documents!"})

    # Greeting responses
    greetings = ["hi", "hello", "hey", "yo", "sup", "wassup!", "what's up"]
    greeting_responses = [
        "Hey there! How can I help you with your documents today?",
        "Welcome! Ready to explore your documents?",
        "Hi! Got any questions about your files?",
        "Wassup!, wanna brainstorm with your files? Ask anything from it and I'll reply ;)",
        "Hi! I am ScandDoc AI, your local document reading assistant."
    ]
    if user_input_lower in greetings:
        return jsonify({"response": random.choice(greeting_responses)})

    # Appreciation responses
    appreciation_keywords = [
        "cool", "you are cool", "damn", "you are smart", "smart",
        "good work", "good work!", "excellent", "good job", "funny"
    ]
    appreciation_responses = [
        "Thanks! I appreciate that you liked my work.",
        "Grateful! Let me know if I can do more.",
        "Glad you're impressed — documents are my jam!"
    ]
    if any(word in user_input_lower for word in appreciation_keywords):
        return jsonify({"response": random.choice(appreciation_responses)})

    # Irrelevant question handling
    irrelevant_keywords = [
        "who is", "tell me about", "joke",
        "weather", "tell me a joke", "wanna hear a joke"
    ]
    if any(keyword in user_input_lower for keyword in irrelevant_keywords):
        return jsonify({"response": "I'm your document assistant — not a chatbot just yet. Upload a file and let's explore what's inside."})

    # If relevant, process with agent
    try:
        response = agent_executor.invoke({"input": user_input})
        return jsonify({"response": response["output"]})
    except Exception as e:
        if rag_chain:
            try:
                fallback = rag_chain(user_input)
                return jsonify({"response": fallback})
            except Exception as e2:
                return jsonify({"response": f"Fallback also failed: {str(e2)}"})
        return jsonify({"response": f"Agent Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
