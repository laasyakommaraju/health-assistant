import os
import ssl
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from Bio import Entrez

# --- SSL CERTIFICATE FIX ---
# This block handles potential SSL certificate verification errors.
# It tells Python to use a less strict SSL context, which is often necessary
# on corporate or university networks.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
# --- END SSL FIX ---

# --- CONFIGURATION ---
load_dotenv()
Entrez.email = "kommarajulaasya@gmail.com" # IMPORTANT: Change this to your email
PINECONE_INDEX_NAME = "health-assistant-rag-gemini"

def fetch_pubmed_articles(query, max_results=15):
    print(f"Fetching PubMed articles for query: '{query}'...")
    # (Function content is unchanged)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    if not id_list:
        print(f"No articles found for query: '{query}'.")
        return []
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    documents = []
    for record in records.get('PubmedArticle', []):
        try:
            article_title = record['MedlineCitation']['Article']['ArticleTitle']
            abstract = record['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
            pubmed_id = record['MedlineCitation']['PMID']
            content = f"Title: {article_title}\n\nAbstract: {abstract}"
            doc = Document(
                page_content=content,
                metadata={"source": "pubmed", "pubmed_id": str(pubmed_id), "query": query}
            )
            documents.append(doc)
        except KeyError:
            continue
    print(f"Fetched {len(documents)} articles from PubMed.")
    return documents

def load_synthetic_data():
    print("Loading synthetic drug and patient data...")
    # (Function content is unchanged)
    documents = []
    drug_data = [
        {"name": "Metformin", "uses": "Treats type 2 diabetes.", "side_effects": "Nausea, diarrhea."},
        {"name": "Lisinopril", "uses": "Treats high blood pressure.", "side_effects": "Dizziness, cough."}
    ]
    for drug in drug_data:
        content = f"Drug Name: {drug['name']}\nUses: {drug['uses']}\nSide Effects: {drug['side_effects']}"
        documents.append(Document(page_content=content, metadata={"source": "drug_database", "drug_name": drug['name']}))

    patient_data = [
        {"id": "patient-001", "diagnosis": "Type 2 Diabetes, Hypertension", "medications": "Metformin, Lisinopril", "notes": "Patient reports dry cough."},
        {"id": "patient-002", "diagnosis": "Chronic Kidney Disease", "medications": "None", "notes": "GFR is stable."}
    ]
    for patient in patient_data:
        content = f"Patient ID: {patient['id']}\nDiagnosis: {patient['diagnosis']}\nMedications: {patient['medications']}\nNotes: {patient['notes']}"
        documents.append(Document(page_content=content, metadata={"source": "patient_records", "patient_id": patient['id']}))
    print(f"Loaded {len(documents)} synthetic documents.")
    return documents

def main():
    # 1. Load data
    all_documents = []
    all_documents.extend(fetch_pubmed_articles("type 2 diabetes management"))
    all_documents.extend(fetch_pubmed_articles("hypertension new treatments"))
    all_documents.extend(load_synthetic_data())

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_documents)
    print(f"\nSplit {len(all_documents)} documents into {len(chunked_docs)} chunks.")

    # 3. Initialize Embeddings and Pinecone
    print("\nInitializing Google Gemini embeddings and Pinecone...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not google_api_key or not pinecone_api_key:
        raise ValueError("GOOGLE_API_KEY or PINECONE_API_KEY not found in environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    pc = Pinecone(api_key=pinecone_api_key)

    # 4. Create or connect to index
    print(f"\nChecking for Pinecone index '{PINECONE_INDEX_NAME}'...")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new index '{PINECONE_INDEX_NAME}' with dimension 768...")
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=768, metric='cosine', spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
        print("Index created successfully.")
    else:
        print("Index already exists.")

    # 5. Upload data
    print("\nUploading document chunks and embeddings to Pinecone...")
    LangchainPinecone.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print("\n--- Data ingestion and indexing complete! ---")

if __name__ == "__main__":
    main()


