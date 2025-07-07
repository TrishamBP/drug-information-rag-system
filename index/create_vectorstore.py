import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from index.vectorstore import DrugVectorStore


def create_production_vectorstore():
    """Create the full production vector store"""
    
    print("🚀 Creating Production Drug Vector Store")
    print("=" * 60)
    
    # Check if data exists
    jsonl_path = "data/processed/fda_documents.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} not found.")
        print("Please run the data ingestion first: python ingest/drug_ingest.py")
        return False
    
    # Create production vector store
    print("🔧 Initializing vector store...")
    vector_store = DrugVectorStore(
        db_name="drug_vector_db",          # Production database name
        embedding_model="openai",          # Using OpenAI embeddings
        chunk_size=1000,                   # Production chunk size
        chunk_overlap=200                  # Production overlap
    )
    
    # Load all documents
    print("📁 Loading all FDA drug documents...")
    start_time = time.time()
    
    documents = vector_store.load_documents_from_jsonl(jsonl_path)
    load_time = time.time() - start_time
    
    print(f"✅ Loaded {len(documents):,} documents in {load_time:.1f}s")
    
    # Create vector store (this will take several minutes with all documents)
    print("🧠 Creating embeddings and building vector store...")
    print("⏳ This may take 5-15 minutes depending on your internet connection...")
    
    embedding_start = time.time()
    vector_store.create_vectorstore(documents)
    embedding_time = time.time() - embedding_start
    
    print(f"✅ Vector store created in {embedding_time:.1f}s")
    
    # Get final statistics
    print("\n📊 Production Vector Store Statistics:")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test the production vector store
    print("\n🔍 Testing production vector store...")
    test_queries = [
        "diabetes medication",
        "blood pressure tablet", 
        "antibiotic injection",
        "pain relief capsule"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        results = vector_store.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            form = doc.metadata.get('form', 'Unknown')
            print(f"  {i}. {drug_name} ({form})")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Production vector store created successfully!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"💾 Database location: drug_vector_db/")
    
    return True


def main():
    """Main function"""
    try:
        success = create_production_vectorstore()
        if success:
            print("\n" + "=" * 60)
            print("✅ PRODUCTION VECTOR STORE READY!")
            print("🚀 You can now use this for your RAG pipeline")
            print("📁 Database: drug_vector_db/")
            print("=" * 60)
        else:
            print("\n❌ Failed to create production vector store")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Vector store creation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 