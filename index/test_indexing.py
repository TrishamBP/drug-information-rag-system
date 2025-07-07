import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from index.vectorstore import DrugVectorStore
from index.embed import VectorVisualizer


def test_data_loading():
    """Test loading documents from JSONL"""
    print("🔍 Testing data loading...")
    
    jsonl_path = "data/processed/fda_documents.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} not found. Please run drug_ingest.py first.")
        return False
    
    # Check first few lines
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(3)]
    
    print(f"✅ Found {jsonl_path}")
    print(f"📄 Sample document:")
    
    try:
        sample_doc = json.loads(lines[0])
        for key, value in list(sample_doc.items())[:5]:
            print(f"  {key}: {str(value)[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Error parsing JSONL: {e}")
        return False


def test_vector_store_creation():
    """Test creating vector store with small sample"""
    print("\n🔧 Testing vector store creation...")
    
    try:
        # Create vector store with OpenAI embeddings
        print("Creating vector store with OpenAI embeddings...")
        vector_store = DrugVectorStore(
            db_name="test_drug_vector_db",
            embedding_model="openai",  # Use OpenAI embeddings
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Load documents (limit to first 100 for testing)
        jsonl_path = "data/processed/fda_documents.jsonl"
        all_documents = vector_store.load_documents_from_jsonl(jsonl_path)
        
        # Use subset for testing
        test_documents = all_documents[:100]
        print(f"📊 Using {len(test_documents)} documents for testing")
        
        # Create vector store
        vector_store.create_vectorstore(test_documents)
        
        # Get stats
        stats = vector_store.get_stats()
        print(f"📈 Vector Store Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None


def test_similarity_search(vector_store):
    """Test similarity search functionality"""
    print("\n🔍 Testing similarity search...")
    
    try:
        # Test queries
        test_queries = [
            "antibiotic tablet",
            "heart medication",
            "pain relief",
            "blood pressure"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            results = vector_store.similarity_search(query, k=3)
            
            for i, doc in enumerate(results, 1):
                drug_name = doc.metadata.get('drug_name', 'Unknown')
                form = doc.metadata.get('form', 'Unknown')
                print(f"  {i}. {drug_name} ({form})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        return False


def test_visualization(vector_store):
    """Test visualization creation"""
    print("\n📊 Testing visualization...")
    
    try:
        visualizer = VectorVisualizer(vector_store)
        
        # Create 2D visualization
        print("Creating 2D t-SNE visualization...")
        reduced_vectors, documents, metadatas = visualizer.create_tsne_visualization(n_components=2)
        
        print(f"✅ Reduced {len(reduced_vectors)} vectors to 2D")
        print(f"📐 Vector shape: {reduced_vectors.shape}")
        
        # Create plot
        fig = visualizer.plot_2d_scatter(reduced_vectors, metadatas, documents, color_by="form")
        
        # Save visualization
        test_output = "test_visualization.html"
        visualizer.save_visualization(fig, test_output)
        
        # Analyze clusters
        df = visualizer.analyze_clusters(reduced_vectors, metadatas)
        print(f"📊 Analysis DataFrame shape: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    test_files = [
        "test_drug_vector_db",
        "test_visualization.html"
    ]
    
    for file_path in test_files:
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
                print(f"✅ Removed {file_path}")
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")


def main():
    """Run all tests"""
    print("🧪 Running Drug RAG Indexing Tests")
    print("=" * 50)
    
    # Test 1: Data loading
    if not test_data_loading():
        return False
    
    # Test 2: Vector store creation
    vector_store = test_vector_store_creation()
    if not vector_store:
        return False
    
    # Test 3: Similarity search
    if not test_similarity_search(vector_store):
        return False
    
    # Test 4: Visualization
    if not test_visualization(vector_store):
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("🚀 Your indexing pipeline is working correctly!")
    
    # Clean up
    cleanup_test_files()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_test_files()
        sys.exit(1) 