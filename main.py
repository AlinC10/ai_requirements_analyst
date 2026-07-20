from vector_database import VectorDatabase
from rag_system import RagSystem

# test current features
db = VectorDatabase()
pr = RagSystem()

db.add_documents("pdf/test.pdf")

while True:
    prompt = input("Prompt? CTRL+C for exit")
    pr.qa_prompt(prompt)

