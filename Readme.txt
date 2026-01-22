Healthcare Agentic AI System



A simple Python-based Healthcare Memory Agent that stores and retrieves a user's medical facts (diagnosis, allergies, medications) using vector similarity search. The agent can recall relevant past medical information when a user asks a health-related question based on symptoms.

This project demonstrates the core ideas behind Retrieval-Augmented Generation (RAG) using a vector database.



 Features

Store medical facts (diagnosis, allergy, medication)

Retrieve relevant past medical history using semantic similarity

Vector database powered by Qdrant

Sentence embeddings using Sentence Transformers

Simple CLI-based interaction

Clean modular architecture





How It Works 

1. User stores medical facts (e.g., "I have low Vitamin D")


2. Facts are converted into embeddings and stored in Qdrant


3. When the user enters symptoms, they are embedded as a query


4. The agent retrieves semantically similar medical facts


5. Relevant history is shown along with a disclaimer





 Project Structure

Healthcare_Agent
│
├── main.py                 # CLI entry point
├── agent.py                # Core healthcare agent logic
│
├── memory/
│   ├── qdrant_store.py     # Vector DB storage & retrieval
│
├── qdrant_data/            # Local Qdrant storage (auto-created)
└── README.md               # Project documentation



 Requirements

Make sure you have Python 3.9+ installed.

Install dependencies:

pip install qdrant-client sentence-transformers



 How to Run

Navigate to the project directory:

cd Healthcare_Agent

Run the program:

python3 main.py




 
