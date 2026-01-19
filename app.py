import streamlit as st
import weaviate
import os
import re
import base64
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import atexit
from typing import List, Optional

load_dotenv()

st.set_page_config(page_title="Chaos RAG Dashboard", page_icon="⋆⭒˚.⋆", layout="wide")

st.markdown("""
<style>
    .stButton > button {
        background-color: #FF69B4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stChatInput {
        border: 2px solid #FF69B4 !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

def encode_image(image_file) -> str:
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

def find_image_path(page_num: int, img_path: Optional[str] = None) -> Optional[str]:
    if img_path and os.path.exists(img_path):
        return img_path
    base_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "images"),
        os.path.join("data", "processed", "images"),
        "data/processed/images",
        "../data/processed/images",
    ]
    page_num_int = int(page_num) if page_num else 0
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for img_idx in range(10):
            test_path = os.path.join(base_dir, f"page_{page_num_int}_img_{img_idx}.png")
            if os.path.exists(test_path):
                return test_path
    return None

def get_history_string(messages: List[dict], limit: int = 10) -> str:
    chat_history_str = ""
    history_slice = messages[:-1][-limit:] if len(messages) > 1 else []
    for msg in history_slice:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        content = re.sub(r'!?\[.*?\]\(https?://[^\s)]+\)|https?://[^\s)]+', '', content)
        content = content.split("Here is a relevant image:")[0]
        content = content.replace("\n\n---\n\n", " ")
        chat_history_str += f"{role}: {content.strip()}\n"
    return chat_history_str

@st.cache_resource
def get_weaviate_client():
    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    
    try:
        client = weaviate.connect_to_custom(
            http_host=weaviate_host,
            http_port=8080,     
            http_secure=False,
            grpc_host=weaviate_host,
            grpc_port=50051,    
            grpc_secure=False
        )
        return client
    except Exception as e:
        print(f"Connection failed: {e}. Fallback to embedded.")
        return weaviate.connect_to_embedded()

@st.cache_resource
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def is_formula_query(query: str) -> bool:
    formula_keywords = [
        "equation", "formula", "formulas", "equations", 
        "mathematical", "derive", "expression", "derivation"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in formula_keywords)

def is_latex_content(text: str) -> bool:
    latex_indicators = [
        r'\frac', r'\sum', r'\int', r'\partial', r'\sigma', r'\rho',
        r'\beta', r'\alpha', r'\theta', r'\Delta', '$$', '$',
        r'\times', r'\cdot', r'\pm', r'\leq', r'\geq', r'd^2', r'dt^2'
    ]
    return any(indicator in text for indicator in latex_indicators)

def resolve_image_path(props: dict, metadata: dict) -> Optional[str]:
    image_path = props.get("image_path")
    if image_path:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        full_image_path = os.path.join(base_dir, image_path)
        full_image_path = os.path.normpath(full_image_path)
        return full_image_path if os.path.exists(full_image_path) else image_path
    if metadata.get("type") == "image":
        page_num = int(metadata.get("page", 0))
        if page_num > 0:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "images")
            possible_paths = [
                os.path.join(base_dir, f"page_{page_num}_img_0.png"),
                os.path.join(base_dir, f"page_{page_num}_img_1.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
    return None

def search_formulas(query: str, collection, embeddings, k: int = 3) -> List[Document]:
    query_vector = embeddings.embed_query(query)
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        limit=k*3,
        alpha=0.7,
        return_metadata=["distance", "score"]
    )
    formula_docs = []
    for obj in response.objects:
        props = obj.properties
        content = props.get("text", "")
        if is_latex_content(content):
            metadata = {
                "page": props.get("page", 0),
                "type": "formula",
                "source": props.get("source", "chaos_book.pdf"),
            }
            if hasattr(obj, "metadata") and obj.metadata:
                if hasattr(obj.metadata, "distance"):
                    metadata["distance"] = obj.metadata.distance
                if hasattr(obj.metadata, "score"):
                    metadata["score"] = obj.metadata.score
            formula_docs.append(Document(page_content=content, metadata=metadata))
            if len(formula_docs) >= k:
                break
    return formula_docs

def hybrid_search(query: str, k: int = 10, alpha: float = 0.6) -> List[Document]:
    client = get_weaviate_client()
    embeddings = get_embeddings_model()
    cross_encoder = get_cross_encoder()
    query_vector = embeddings.embed_query(query)
    collection = client.collections.get("ChaosKnowledgeBase")
    
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        limit=k,
        alpha=alpha,
        return_metadata=["distance", "score"]
    )
    
    docs = []
    for obj in response.objects:
        props = obj.properties
        metadata = {
            "page": props.get("page", 0),
            "type": props.get("type", "text"),
            "source": props.get("source", "chaos_book.pdf"),
        }
        metadata["image_path"] = resolve_image_path(props, metadata)
        if hasattr(obj, "metadata") and obj.metadata:
            if hasattr(obj.metadata, "distance"):
                metadata["distance"] = obj.metadata.distance
            if hasattr(obj.metadata, "score"):
                metadata["score"] = obj.metadata.score
        content = props.get("text", "")
        docs.append(Document(page_content=content, metadata=metadata))
    
    # Additional formula search
    if is_formula_query(query):
        formula_docs = search_formulas(query, collection, embeddings)
        existing_pages = {doc.metadata.get("page") for doc in docs}
        for fdoc in formula_docs:
            if fdoc.metadata.get("page") not in existing_pages:
                docs.append(fdoc)
    
    # Re-rank 
    if docs:
        pairs = [(query, doc.page_content) for doc in docs[:10]]
        scores = cross_encoder.predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs[:10]), key=lambda x: x[0], reverse=True)]
        return ranked_docs[:3]
    
    return docs[:3]

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, max_tokens=1024)

def create_system_prompt(context_text: str, chat_history_text: str) -> str:
    return f"""
# Role
You are an expert assistant specializing in chaos theory, with deep knowledge from "The Essence of Chaos" by Edward Lorenz. 
Your primary role is to help users understand complex concepts from this book using the provided context.

# Instructions

## Answering Guidelines:
1. **Primary Source**: Base your answers exclusively on the provided **Context** from the book. Do not use external knowledge beyond what's in the context.
2. **Completeness**: If the context contains sufficient information, provide a comprehensive answer. If information is partial, acknowledge what you can answer and what's missing.
3. **Uncertainty**: If the context does not contain enough information to answer the question, clearly state: "This information is not found in the provided context from the book."
4. **Citations**: Always cite page numbers when referencing information. Use the format: [Page X], [Page X, Y, etc.] or [Pages X-Y] for multi-page references.

## Image Handling:
1. **User-Uploaded Images**: If the user provides an image, carefully analyze it and:
   - Describe what you see in the image
   - Compare and relate it to concepts, diagrams, or formulas mentioned in the book context
   - Identify any connections to chaos theory principles from the book
   - Cite relevant pages that discuss related concepts

2. **Context Images/Diagrams**: If the user asks to display/show a diagram, chart, or figure from the book:
   - Acknowledge their request
   - Explain what the diagram represents and its significance
   - Reference the specific page number where it appears
   - The image will be automatically displayed in your response

3. **Mathematical Formulas/Equations**: If the user asks to provide/explain a formula or equation from the book:
   - Provide a clear, step-by-step explanation of the formula
   - Display the formula in proper LaTeX format using $$...$$ for block equations or $...$ for inline equations
   - Explain what each variable represents
   - Reference the specific page number where it appears
   - If relevant, explain the physical or mathematical significance of the equation

## Conversation Continuity:
Use the conversation history to:
- Maintain context across multiple questions
- Understand follow-up questions and clarifications
- Build upon previous explanations
- Avoid repeating information already discussed

## Response Style:
- Write in a clear, educational, and engaging manner
- Use appropriate technical terminology but explain complex concepts simply
- Structure longer answers with clear paragraphs or bullet points when helpful
- For complex topics, you may provide longer explanations, but always prioritize clarity over length

# Context from Book:
{context_text}

# Conversation History:
{chat_history_text}
"""

def generate_response(prompt: str, context_text: str, chat_history: str, uploaded_file=None) -> str:
    llm = get_llm()
    system_prompt = create_system_prompt(context_text, chat_history)
    
    content_blocks = [{"type": "text", "text": f"Context from Book:\n{context_text}\n\nUser Question: {prompt}"}]
    
    if uploaded_file:
        img_base64 = encode_image(uploaded_file)
        content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
    
    message = [SystemMessage(content=system_prompt), HumanMessage(content=content_blocks)]
    return llm.invoke(message).content

def render_sidebar():
    with st.sidebar:
        st.title("⋆˚࿔")
        uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded:
            st.image(uploaded, caption="Image", width="stretch")
            st.success("Image was successfully added")
        st.divider()
        if st.button("Start new session"):
            st.session_state.messages = []
            st.rerun()
        return uploaded

def render_search_analytics(docs: List[Document]):
    with st.expander("Search Analytics", expanded=False):
        data = []
        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get("type", "text")
            type_display = "FORMULA" if doc_type == "formula" else doc_type.upper()
            data.append({"Rank": i+1, "Page": doc.metadata.get("page", 0), "Type": type_display, "Snippet": doc.page_content[:50]+"..."})
        df = pd.DataFrame(data)
        color_map = {"TEXT": "#636EFA", "IMAGE": "#EF553B", "FORMULA": "#00CC96"}
        fig = px.scatter(df, x="Page", y="Rank", color="Type", size=[10]*len(df), color_discrete_map=color_map)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

def render_sources(docs: List[Document]):
    with st.expander("📚 Sources from Book", expanded=False):
        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get("type", "text")
            icon = "📐" if doc_type=="formula" else ("🖼️" if doc_type=="image" else "📄")
            page_num = doc.metadata.get("page",0)
            st.markdown(f"**{icon} Source {i+1} (Page {page_num})**")
            if doc_type=="image":
                img_path = find_image_path(page_num, doc.metadata.get("image_path"))
                if img_path:
                    st.image(img_path, caption=f"Figure from Page {page_num}", width=400)
                else:
                    st.warning(f"Image file not found for page {page_num}")
                if doc.page_content:
                    st.caption(f"*{doc.page_content}*")
            elif doc_type=="formula":
                st.markdown(doc.page_content)
            else:
                if doc.page_content:
                    st.caption(doc.page_content[:300] + ("..." if len(doc.page_content)>300 else ""))
                else:
                    st.warning("No content available for this document.")
            if i < len(docs)-1:
                st.divider()

def render_first_image(docs: List[Document]):
    for doc in docs:
        if doc.metadata.get("type")=="image":
            img_path = find_image_path(doc.metadata.get("page"), doc.metadata.get("image_path"))
            if img_path:
                st.image(img_path, caption=f"Figure from Page {doc.metadata.get('page')}", width=500)
                break


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def handle_user_input(prompt: str, uploaded_file):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file:
            st.image(uploaded_file, width=200)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            placeholder.markdown("Thinking...")
            chat_history_text = get_history_string(st.session_state.messages)
            with st.spinner("🔍 Searching in the book..."):
                docs = hybrid_search(query=prompt)
            if docs:
                render_search_analytics(docs)
            context_text = "\n\n".join([f"[Page {d.metadata.get('page')}] {d.page_content}" for d in docs])
            full_response = generate_response(prompt, context_text, chat_history_text, uploaded_file)
            placeholder.markdown(full_response)
            render_first_image(docs)
            render_sources(docs)
            st.session_state.messages.append({"role":"assistant","content":full_response})
        except Exception as e:
            placeholder.error(f"Error: {e}")
            st.session_state.messages.append({"role":"assistant","content":f"Error: {e}"})

def run_app():
    initialize_session_state()
    st.title("Chaos Assistant")
    uploaded_file = render_sidebar()
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    if prompt := st.chat_input("Ask about chaos theory"):
        handle_user_input(prompt, uploaded_file)

def cleanup():
    client = get_weaviate_client()
    try:
        if client and hasattr(client,'is_ready') and client.is_ready():
            client.close()
    except:
        pass

atexit.register(cleanup)

if __name__=="__main__":
    run_app()
