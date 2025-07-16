from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate.from_template("""
You are ScandDoc AI, a helpful assistant that answers questions based on the provided documents.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the context above.
- If unsure, say: "I don't have that info in the documents."
- Be specific. Mention documents if possible.

Answer:
""")

agent_prompt = PromptTemplate.from_template("""
You are ScandDoc AI, a document question-answering agent.

Available tools:
{tools}

Use this format:
Question: {input}
Thought: [what you think]
Action: [tool_name]
Action Input: [string input]
Observation: [result]
... (repeat as needed)
Final Answer: [your answer]

Question: {input}
Thought: {agent_scratchpad}
""")
