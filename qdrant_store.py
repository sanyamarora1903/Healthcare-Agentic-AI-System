from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

COLLECTION_NAME = "medical_memory"

client = QdrantClient(path="./qdrant_data")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


def store_fact(fact: str, fact_type: str):
    vector = embedding_model.encode(fact).tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": fact,
                    "type": fact_type
                }
            )
        ]
    )


def retrieve_facts(query: str, limit: int = 2):
    query_vector = embedding_model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )

    facts = []
    for point in results.points:  # iterate over dicts
        if point.payload and "text" in point.payload:
            facts.append(point.payload["text"])



    return facts
