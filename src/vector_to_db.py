import weaviate
from dotenv import load_dotenv
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chunk_split import PrepareData

load_dotenv()

def embeddings_to_bd():
    chunker = PrepareData()
    print("Loading and chunking PDF...") 
    docs = chunker.load_and_chunk()
    print(f"Prepared {len(docs)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    client = weaviate.connect_to_local(
        port=8079,
        grpc_port=50050
    )

    try:        
        print("Connecting to Weaviate and uploading vectors...")
        WeaviateVectorStore.from_documents(
            client=client,
            documents=docs,
            embedding=embeddings,
            index_name="ChaosKnowledgeBase" 
        )
        
        print("Data uploaded to Docker Weaviate.")
        
    finally:
        client.close()

if __name__ == "__main__":
    embeddings_to_bd()





