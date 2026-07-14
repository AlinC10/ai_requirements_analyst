from vector_database import VectorDatabase
from rag_system import Prompts

# test current features
db = VectorDatabase()
pr = Prompts()

db.add_documents("pdf/test.pdf")

while True:
    prompt = input("Prompt? CTRL+C for exit")
    pr.qa_prompt(prompt)

