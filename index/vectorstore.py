import os
import json
from typing import List, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(override=True)

class DrugVectorStore:
    """Vector store for FDA drug documents using ChromaDB"""
    
    def __init__(self, 
                 db_name: str = "drug_vector_db",
                 embedding_model: str = "openai",  # "openai" or "huggingface"
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        if embedding_model == "openai":
            # Set up OpenAI API key
            os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
            self.embeddings = OpenAIEmbeddings()
            print("Using OpenAI embeddings")
        else:
            # Use free HuggingFace embeddings
            try:
                # Try the standard model first
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                print("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Falling back to basic model...")
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="paraphrase-MiniLM-L6-v2"
                    )
                    print("Using HuggingFace embeddings (paraphrase-MiniLM-L6-v2)")
                except Exception as e2:
                    raise ValueError(f"Could not load any HuggingFace embedding model. Error: {e2}")
        
        # Initialize text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        self.vectorstore = None
    
    def load_documents_from_jsonl(self, jsonl_path: str) -> List[Document]:
        """Load FDA drug documents from JSONL file"""
        documents = []
        
        print(f"Loading documents from {jsonl_path}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Create document content from drug data
                    content = self._create_document_content(data)
                    
                    # Create metadata
                    metadata = {
                        "doc_type": "fda_drug",
                        "application_no": data.get("application_no", ""),
                        "product_no": data.get("product_no", ""),
                        "drug_name": data.get("drug_name", ""),
                        "form": data.get("form", ""),
                        "marketing_status": data.get("marketing_status", ""),
                        "sponsor_name": data.get("sponsor_name", ""),
                        "source": "FDA"
                    }
                    
                    # Create LangChain Document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def _create_document_content(self, data: dict) -> str:
        """Create searchable content from drug data"""
        content_parts = []
        
        # Add drug name and active ingredient
        if data.get("drug_name"):
            content_parts.append(f"Drug Name: {data['drug_name']}")
        
        if data.get("active_ingredient"):
            content_parts.append(f"Active Ingredient: {data['active_ingredient']}")
        
        # Add form and strength
        if data.get("form"):
            content_parts.append(f"Form: {data['form']}")
        
        if data.get("strength"):
            content_parts.append(f"Strength: {data['strength']}")
        
        # Add regulatory information
        if data.get("marketing_status"):
            content_parts.append(f"Marketing Status: {data['marketing_status']}")
        
        if data.get("application_type"):
            content_parts.append(f"Application Type: {data['application_type']}")
        
        if data.get("te_code"):
            content_parts.append(f"Therapeutic Equivalence Code: {data['te_code']}")
        
        # Add submission information
        if data.get("submission_type"):
            content_parts.append(f"Submission Type: {data['submission_type']}")
        
        if data.get("submission_status"):
            content_parts.append(f"Submission Status: {data['submission_status']}")
        
        # Add sponsor
        if data.get("sponsor_name"):
            content_parts.append(f"Sponsor: {data['sponsor_name']}")
        
        # Add description if available
        if data.get("description"):
            content_parts.append(f"Description: {data['description']}")
        
        return "\n".join(content_parts)
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create vector store from documents"""
        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Delete existing database if it exists
        if os.path.exists(self.db_name):
            print(f"Deleting existing vector store: {self.db_name}")
            try:
                existing_store = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
                existing_store.delete_collection()
            except Exception as e:
                print(f"Error deleting existing collection: {e}")
        
        # Create new vectorstore
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings, 
            persist_directory=self.db_name
        )
        
        count = self.vectorstore._collection.count()
        print(f"✅ Vector store created with {count:,} documents")
        
        # Get sample embedding info
        try:
            sample_embedding = self.vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
            dimensions = len(sample_embedding)
            print(f"📊 Vectors: {count:,} documents with {dimensions:,} dimensions each")
        except Exception as e:
            print(f"Could not get embedding dimensions: {e}")
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store"""
        if os.path.exists(self.db_name):
            print(f"Loading existing vector store from {self.db_name}")
            self.vectorstore = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
            count = self.vectorstore._collection.count()
            print(f"Loaded vector store with {count:,} documents")
            return self.vectorstore
        else:
            print(f"No existing vector store found at {self.db_name}")
            return None
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the vector store instance"""
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() or load_vectorstore() first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get unique document types
            result = collection.get(include=['metadatas'])
            doc_types = set()
            drug_names = set()
            sponsors = set()
            
            for metadata in result['metadatas']:
                if metadata.get('doc_type'):
                    doc_types.add(metadata['doc_type'])
                if metadata.get('drug_name'):
                    drug_names.add(metadata['drug_name'])
                if metadata.get('sponsor_name'):
                    sponsors.add(metadata['sponsor_name'])
            
            return {
                "total_documents": count,
                "document_types": list(doc_types),
                "unique_drugs": len(drug_names),
                "unique_sponsors": len(sponsors),
                "database_path": self.db_name
            }
        except Exception as e:
            return {"error": f"Could not get stats: {e}"}


def main():
    """Main function for testing"""
    # Create vector store
    vector_store = DrugVectorStore(
        db_name="drug_vector_db",
        embedding_model="openai",  # Use OpenAI embeddings
        chunk_size=800,
        chunk_overlap=100
    )
    
    # Load documents
    jsonl_path = "data/processed/fda_documents.jsonl"
    documents = vector_store.load_documents_from_jsonl(jsonl_path)
    
    # Create vector store
    vector_store.create_vectorstore(documents)
    
    # Print stats
    stats = vector_store.get_stats()
    print(f"\n📈 Vector Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test search
    print(f"\n🔍 Testing similarity search...")
    results = vector_store.similarity_search("amoxicillin antibiotic", k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Drug: {doc.metadata.get('drug_name', 'N/A')}")
        print(f"  Form: {doc.metadata.get('form', 'N/A')}")
        print(f"  Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main() 