import os
from dotenv import load_dotenv
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunk_split import PrepareData
import weaviate
import time
from weaviate.classes.init import Auth

load_dotenv()

def embeddings_to_bd():
    chunker = PrepareData()
    print("Loading and chunking PDF...") 
    docs = chunker.load_and_chunk()
    print(f"Prepared {len(docs)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    try:        
        print("Connecting to Weaviate Cloud and uploading vectors...")

        for i in range(0, len(docs), 50):
            batch = docs[i:i+50]
            print(f"batch: {i//50 + 1}")

            if i == 0:
                WeaviateVectorStore.from_documents(
                   client=client,
                   documents=batch,
                   embedding=embeddings,
                   index_name="ChaosKnowledgeBase",
                   text_key="text"
                )
            else:
                vector_store = WeaviateVectorStore(
                    client=client,
                    index_name="ChaosKnowledgeBase",
                    text_key="text",
                    embedding=embeddings
                )
                vector_store.add_documents(batch)
            
            if i + 50 < len(docs):
                time.sleep(60)

        print("Data uploaded to Weaviate Cloud.")
        
    finally:
        client.close()

if __name__ == "__main__":
    embeddings_to_bd()