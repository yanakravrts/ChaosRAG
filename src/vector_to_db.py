import os
import weaviate
from dotenv import load_dotenv
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunk_split import PrepareData

load_dotenv()

def embeddings_to_bd():
    chunker = PrepareData()
    docs = chunker.load_and_chunk()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    client = weaviate.connect_to_embedded()

    try:        
        vector_store = WeaviateVectorStore.from_documents(
            client=client,
            documents=docs,
            embedding=embeddings,
            index_name="ChaosKnowledgeBase" 
        )
        
        print("done")
        
    finally:
        client.close()

if __name__ == "__main__":
    embeddings_to_bd()







