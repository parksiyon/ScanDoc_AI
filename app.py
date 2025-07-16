from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from backend.agent import agent_executor
from backend.pipeline import rag_chain, vector_db
import os

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
        return jsonify({"response": " Vector database not initialized."})
    user_input = request.json.get("query", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid question."})
    # Handle common small talk or irrelevant inputs
    greetings = ["hi", "hello", "hey", "yo", "sup", "wassup!", "what's up!"]
    if user_input.lower() in greetings:
        return jsonify({"response": " Hey there! How can I help you with your documents today?"})
    irrelevant_keywords = ["who is", "tell me about", "what is your name", "joke", "funny", "weather", "tell me a joke"]
    if any(kw in user_input.lower() for kw in irrelevant_keywords):
        return jsonify({"response": " I'm your document assistant — not a chatbot just yet. Upload a file and let's explore what's inside!"})
    try:
        response = agent_executor.invoke({"input": user_input})
        return jsonify({"response": response["output"]})
    except Exception as e:
        if rag_chain:
            try:
                fallback = rag_chain(user_input)
                return jsonify({"response": fallback})
            except Exception as e2:
                return jsonify({"response": f" Fallback also failed: {str(e2)}"})
        return jsonify({"response": f" Agent Error: {str(e)}"})
if __name__ == "__main__":
    app.run(debug=True)


