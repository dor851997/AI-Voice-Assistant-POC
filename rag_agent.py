import logging
import os
# Use OpenAI's embedding model.
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import datetime
import uuid

# -------------------------------
# RAGAgent: Saves interactions for retrieval-augmented generation using Pinecone
# -------------------------------
class RAGAgent:
    def __init__(self) -> None:
        # Initialize a list to store interactions for reference.
        self.interactions: list[dict[str, str]] = []
        self.client = OpenAI()
        self.embedding_model_name = "text-embedding-ada-002"
        # text-embedding-ada-002 returns embeddings of dimension 1536.
        self.dim = 1536
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        # Normalize the environment: if it ends with '-aws', remove that suffix so the region is correct
        if pinecone_environment.endswith("-aws"):
            normalized_region = pinecone_environment.replace("-aws", "")
        else:
            normalized_region = pinecone_environment

        # Create an instance of Pinecone using the new API
        pc = Pinecone(api_key=pinecone_api_key)

        self.index_name = "tv-interactions"

        # Check if the index exists using the new API; create it if it does not
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric='euclidean',
                spec=ServerlessSpec(cloud='aws', region=normalized_region)
            )

        # Obtain the index instance
        self.pinecone_index = pc.Index(self.index_name)

        # List to store interaction texts corresponding to embeddings.
        self.interaction_texts = []

        logging.info(f"Pinecone index '{self.index_name}' initialized with dimension: {self.dim}")

    def add_interaction(self, user_message: str, assistant_message: str) -> None:
        stats = self.pinecone_index.describe_index_stats()
        logging.info(f"Index stats after upsert: {stats}")
        interaction = {"user": user_message, "assistant": assistant_message}
        self.interactions.append(interaction)

        # Concatenate messages.
        combined_text = user_message + " " + assistant_message
        
        try:
            embedding_response = self.client.embeddings.create(model=self.embedding_model_name,
            input=combined_text)
            embedding = embedding_response.data[0].embedding
        except Exception as e:
            logging.error(f"Error obtaining embedding for interaction: {e}")
            return

        # Use a unique ID for the vector.
        vector_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        metadata = {
            "user": user_message,
            "assistant": assistant_message,
            "text": combined_text,
            "timestamp": timestamp
        }        
        upsert_response = self.pinecone_index.upsert(vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}])
        logging.info(f"Upsert response: {upsert_response}")
        self.interaction_texts.append(combined_text)

        logging.info(f"RAG stored interaction in Pinecone with id {vector_id}: {interaction}")

    async def augment(self, prompt: str) -> str:
        """
        Augment the prompt using external context.
        For now, returns the prompt unchanged.
        """
        return prompt

    def search_interactions(self, query: str, top_k: int = 5) -> list[str]:
        """
        Search for similar interactions based on a query.
        
        Args:
            query (str): The input query to search for.
            top_k (int): The number of top matching interactions to return.
            
        Returns:
            list[str]: A list of interaction texts that closely match the query.
        """        
        try:
            embedding_response = self.client.embeddings.create(model=self.embedding_model_name,
            input=query)
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            logging.error(f"Error obtaining embedding for search query: {e}")
            return []

        # Query Pinecone.
        result = self.pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])
        return [match["metadata"]["text"] for match in matches]
