import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import Pinecone as LangchainPinecone

# --- SETUP ---
load_dotenv()

PINECONE_INDEX_NAME = "health-assistant-rag-gemini"

# --- FUNCTIONS ---

@st.cache_resource
def initialize_models():
    """Initializes and caches the language and embedding models."""
    print("Initializing models...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # --- MODEL NAME UPDATED HERE ---
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return embeddings, llm

def setup_rag_chain(_embeddings, _llm):
    """Sets up and caches the RAG chain."""
    vectorstore = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, _embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    template = """
    You are a helpful healthcare assistant. Your purpose is to provide information based on the context provided.
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Use only the following pieces of context to answer the question at the end.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# --- STREAMLIT APP ---

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺")
st.title("🩺 Healthcare Information Assistant")
st.markdown("Ask me anything about the loaded medical documents, patient records, or drug information.")

try:
    embeddings_model, llm_model = initialize_models()
    rag_chain, retriever_model = setup_rag_chain(embeddings_model, llm_model)
except Exception as e:
    st.error(f"Failed to initialize. Please check your API keys and library versions. Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            full_response = ""
            try:
                retrieved_docs = retriever_model.invoke(prompt)
                full_response = rag_chain.invoke(prompt)
                message_placeholder.markdown(full_response)

                with st.expander("Show Sources"):
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get('source', 'N/A')
                        st.write(f"**Source {i+1}:** *({source})*")
                        st.caption(doc.page_content)
            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
